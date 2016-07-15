/*
 * ridge.cpp
 *
 *  Created on: Jan 11, 2016
 *      Author: Chris
 */

#include <stdlib.h>
#include <string.h>
#include <cstring>
#include <sstream>
#include <cmath>
#include "comm.h"
#include "run.h"
#include "domain.h"
#include "update.h"
#include "force.h"
#include "integrate.h"
#include "modify.h"
#include "output.h"
#include "finish.h"
#include "input.h"
#include "timer.h"
#include "error.h"
#include "bisection.h"
#include "read_dump.h"
#include "write_dump.h"
#include "output.h"
#include "memory.h"
#include "min.h"
#include "minimize.h"
#include "atom.h"
#include "fix.h"
#include "fix_store.h"
#include "fix_store_lat.h"
#include "ridge.h"
#include "irregular.h"
#include <iostream>
#include "compute.h"
#include "compute_freq.h"

using namespace LAMMPS_NS;

#define MAXLINE 2048

/* ---------------------------------------------------------------------- */

Ridge::Ridge(LAMMPS *lmp) : Pointers(lmp) {
	//epsF = 25e-3;
	nPRelSteps = 5;
	nMRelSteps = 1000;
}


/* ---------------------------------------------------------------------- */

void Ridge::command(int narg, char **arg){
	
	if(narg<4) error->all(FLERR,"Ridge Method -- Illegal run command");

	nRSteps = force->numeric(FLERR,arg[0]);
	nBSteps = force->numeric(FLERR,arg[1]);
	epsT = force->numeric(FLERR,arg[2]);
	epsF = force->numeric(FLERR,arg[3]);

	PerformRidge();
	
	return;
}

void Ridge::PerformRidge()
{
        int me;
	bool sFlag;
        MPI_Comm_rank(world,&me);
	if(me==0) OpenTLS();
	CallMinimize();
        if(LoadPositions()<0) error->all(FLERR,"Ridge Method -- Atomic positions not stored in fix_store.");
	InitAtomArrays();

	CopyAtoms(atom->x,pTLS1);
	CopyLatToBox(lat1);
	eTLS1 = CallMinimize();
	CopyAtoms(lAtoms,atom->x);
	CopyBoxToLat(lat1);
	CopyBoxToLat(lLat);
	
	CopyAtoms(atom->x,pTLS2);
	CopyLatToBox(lat2);
	eTLS2 = CallMinimize();
	CopyAtoms(hAtoms,atom->x);
	CopyBoxToLat(lat2);
	CopyBoxToLat(hLat);
	float chDist = ComputeDistance(lAtoms,hAtoms);
	if(chDist<epsT)
	{
		if(me==0) fprintf(screen, "UPDATE-End-points relaxed to same minimum (distance is %f), leaving ridge method.\n", chDist);
		return;
	}
	if(me==0) fprintf(screen, "UPDATE-Asymmetry: %f\n",fabs(eTLS2-eTLS1));

	//TestComputeDistance();
	//Runs a couple tests to check that the BisectPositions function is working correctly.
	//TestBisect();	
	for(int i=0; i<nRSteps;i++)
	{
		for(int j=0; j<nBSteps;j++)
		{
			BisectPositions(lAtoms, hAtoms, tAtoms);
			CopyLatToBox(tLat);
			sFlag = CheckSaddle(tAtoms);
			if(sFlag)
			{
				WriteTLS(eTLS1,eTLS2,update->minimize->efinal);
				break;
			}

			ComparePositions(lAtoms, hAtoms, tAtoms);
		}

		if(sFlag) break;
		PartialRelax(lAtoms, hAtoms);
	}

	if(!sFlag) if(me==0) fprintf(screen, "UPDATE-Cannot find saddle.\n");

	if(atom->map_style != 0)
	{
                atom->map_delete();
                atom->map_style = 0;
	}
		
	if(me==0) fclose(fp);
	modify->delete_fix((char *) "TLSt");
	modify->delete_fix((char *) "TLSl");
	modify->delete_fix((char *) "TLSh");
	
	return;
}

void Ridge::BisectPositions(double** pos1, double** pos2, double** posOut)
{
	int m;
	double diff = 0.0;

	//Fills posOut with the values from pos1.  These should be shifted only for local atoms that move between pos1 and pos2.
	CopyAtoms(posOut,pos1);

	//Loops over atoms, using the mapping from atom->map.  If the atom is owned by the processor, its position is shifted by the difference between pos2 and pos1 for that atom.
	for(int i=0;i<atom->nlocal;i++)
	{
		for(int j=0; j<domain->dimension;j++)
		{
			diff = pos2[i][j] - pos1[i][j];
			//This if-statement ensures that the difference is not being computed across the unit cell if the atoms are just moving from one side of the unit cell to the other.
			if(diff < -domain->prd_half[j])
			{
				diff = diff + domain->prd[j];
			}
			else if(diff > domain->prd_half[j])
			{
				diff = diff - domain->prd[j];
			}
			posOut[i][j] = posOut[i][j] + 0.5*diff;
		}
	}

	for(int i=0;i<9;i++)
	{
		tLat[i] = 0.5*(hLat[i]+lLat[i]);
		if(comm->me==0) std::cout << "Bisect: " << i << tLat[i] << std::endl;
	}

	return;
}

int Ridge::LoadPositions()
{
	int me;
        int m;
	double diff;
        MPI_Comm_rank(world,&me);
	//First, get the labels for the fixes for the TLS atom positions.
	int iTLS1 = modify->find_fix((char *) "TLS1");
	int iTLS2 = modify->find_fix((char *) "TLS2");
	int iTLSl1 = modify->find_fix((char *) "TLSLat1");
	int iTLSl2 = modify->find_fix((char *) "TLSLat2");

	//If there are no corresponding fixes, returns -1 to flag the error.
	if((iTLS1<0)||(iTLS2<0)) return -1;

	//Creates a fix according to the stored fix
	FixStore* TLS1 = (FixStore *) modify->fix[iTLS1];
	FixStore* TLS2 = (FixStore *) modify->fix[iTLS2];
	FixStoreLat* TLSl1 = (FixStoreLat *) modify->fix[iTLSl1];
	FixStoreLat* TLSl2 = (FixStoreLat *) modify->fix[iTLSl2];

        if(me==0) std::cout << "Loading Atoms" << std::endl;

	//Copies the array in the FixStore to the arrays used within this class.
	//CopyAtoms(atom->x,TLS1->astore);
	pTLS1 = TLS1->astore;
	pTLS2 = TLS2->astore;
	lat1 = TLSl1->vstore;
	lat2 = TLSl2->vstore;
	for(int i=0; i<9;i++) std::cout << lat1[i] << "\t" << lat2[i] << std::endl;


	//Creates FixStore for Saddle Point configuration
        char **newarg = new char*[5];

	//Created the arguments for the StoreFix
        newarg[0] = (char *) "TLSs";
        newarg[1] = (char *) "all";
        newarg[2] = (char *) "STORE";
        newarg[3] = (char *) "0";
        newarg[4] = (char *) "3";

	//Adds the Fix, and stores the pos1 array in the astore variable of the StoreFix command.
        modify->add_fix(5,newarg);
	int iTLSs = modify->find_fix((char *) "TLSs");
        FixStore *TLSs = (FixStore *) modify->fix[iTLSs];
        CopyAtoms(TLSs->astore,pTLS1);
	pTLSs = TLSs->astore;

        newarg[0] = (char *) "TLSLatS";
        newarg[2] = (char *) "STORELAT";
        modify->add_fix(3,newarg);
        int iTLSlS = modify->find_fix((char *) "TLSLatS");
        FixStoreLat *TLSlS = (FixStoreLat *) modify->fix[iTLSlS];
        latS = TLSlS->vstore;

	MPI_Barrier(world);

	return 0;
}

void Ridge::ReadPositions()
{
	char** readInput = new char*[4+domain->dimension];
	int ni = -1;
        readInput[ni++] = (char *) "TLS1.dump";
        readInput[ni++] = (char *) "0";
        readInput[ni++] = (char *) "x";
	if(domain->dimension>=2) readInput[ni++] = (char *) "y";
	if(domain->dimension>=3) readInput[ni++] = (char *) "z";
        readInput[ni++] = (char *) "replace";
	readInput[ni++] = (char *) "yes";
	ReadDump *bisRead = new ReadDump(lmp);
        bisRead->command(ni, readInput);	
        CopyAtoms(pTLS1, atom->x);

	readInput[0] = (char *) "TLS2.dump";
        bisRead->command(ni, readInput);
        CopyAtoms(pTLS2, atom->x);
	delete bisRead;
        
        return;
}

double** Ridge::InitAtomArray()
{
        double** atomArray = new double*[atom->natoms];
        for(int i=0; i<atom->natoms; i++)
        {
                atomArray[i] = new double[domain->dimension];

        }
        return atomArray;
}

void Ridge::DeleteAtomArray(double** atomArray)
{
        for(int i=0; i<atom->natoms; i++)
        {
                delete atomArray[i];
        }
        delete atomArray;
}

void Ridge::CopyAtoms(double** copyArray, double** templateArray)
{
        int me;
        int m;
	MPI_Comm_rank(world,&me);
        for(int i=0;i<atom->nlocal;i++)
        {
		for(int j=0;j<domain->dimension;j++)
		{
			copyArray[i][j] = templateArray[i][j];
		}
        }
	for (int i = 0; i < atom->nlocal; i++) domain->remap(atom->x[i],atom->image[i]);
	return;
}

void Ridge::OpenTLS()
{
        std::string strFile = "TLS.dump";
        char *charFile = new char[20];
        std::strcpy(charFile,strFile.c_str());
        fp = fopen(charFile,"a");
        return;
}

void Ridge::WriteTLS(double E1, double E2, double E3)
{
        double Asym = fabs(E2 - E1);
	double Barrier= 0.5*((E3-E1)+(E3-E2));
	int me;
	MPI_Comm_rank(world,&me);
        if(me==0) fprintf(fp, "%f\t%f \n", Asym, Barrier);

	char** dumparg = new char*[8];
        dumparg[0] = (char *) "all";
        dumparg[1] = (char *) "atom";
	dumparg[2] = (char *) "TLS.pos";
	dumparg[3] = (char *) "modify";
	dumparg[4] = (char *) "append";
	dumparg[5] = (char *) "yes";
	dumparg[6] = (char *) "scale";
	dumparg[7] = (char *) "no";

	WriteDump* pDump = new WriteDump(lmp);
	update->reset_timestep(0);
	CopyAtoms(atom->x,pTLS1);
	CopyLatToBox(lat1);
	UpdateMapping();
	pDump->command(8,dumparg);
	update->reset_timestep(1);
	CopyAtoms(atom->x,pTLS2);
	CopyLatToBox(lat2);
	UpdateMapping();
	pDump->command(8,dumparg);
	update->reset_timestep(2);
        CopyAtoms(atom->x,pTLSs);
	CopyLatToBox(latS);
        pDump->command(8,dumparg);
	
	delete dumparg;
	delete pDump;

	MPI_Barrier(world);
        return;
}


void Ridge::PartialRelax(double** lAtoms, double** hAtoms)
{       
        char** newarg = new char*[4];
	char* cRelSteps = new char[4];
	int me;
	int m;
	double** atomPtr =atom->x;
	MPI_Comm_rank(world,&me);
	
	std::ostringstream oss;
        oss << nPRelSteps;
        std::string sRelSteps = oss.str();
        std::strcpy(cRelSteps,sRelSteps.c_str());

        newarg[0] = (char *) "0.0";
        newarg[1] = (char *) "1.0e-6";
        newarg[2] = cRelSteps;
        newarg[3] = (char *) "1000";

	Minimize* rMin = new Minimize(lmp);
	CopyAtoms(atomPtr,lAtoms);
        rMin->command(4, newarg);
	CopyAtoms(lAtoms,atomPtr);

        CopyAtoms(atomPtr,hAtoms);
        rMin->command(4, newarg);
        CopyAtoms(hAtoms,atomPtr);

        delete rMin;

        return;
}

void Ridge::ComparePositions(double** lAtoms, double** hAtoms, double** tAtoms)
{
	double lDistDiff, hDistDiff, mDistDiff;
	double tEnergy;
        int me;
        MPI_Comm_rank(world,&me);

	CopyAtoms(atom->x,tAtoms);
	CopyLatToBox(tLat);
	UpdateMapping();
	CallMinimize();
	tEnergy = update->minimize->einitial;

	lDistDiff = ComputeDistance(atom->x, pTLS1);
	hDistDiff = ComputeDistance(atom->x, pTLS2);
	mDistDiff = ComputeDistance(pTLS1, pTLS2);
	if((lDistDiff<epsT) && (lDistDiff<hDistDiff))
	{
		CopyAtoms(lAtoms,tAtoms);
		CopyLatToLat(lLat, tLat);
		if(me==0)  fprintf(screen, "UPDATE-Match L (%f, %f, %f): V1 = %f, V2 = %f \n", lDistDiff, hDistDiff, mDistDiff, tEnergy - eTLS1, tEnergy - eTLS2);
	}
	else if(hDistDiff<epsT)
	{
		CopyAtoms(hAtoms,tAtoms);
		CopyLatToLat(hLat, tLat);
		if(me==0)  fprintf(screen, "UPDATE-Match U (%f, %f, %f): V1 = %f, V2 = %f \n", lDistDiff, hDistDiff, mDistDiff, tEnergy - eTLS1, tEnergy - eTLS2);
	}
	else
	{
		CopyAtoms(hAtoms,tAtoms);
		CopyLatToLat(hLat, tLat);
		CopyAtoms(pTLS2, atom->x);
		CopyBoxToLat(lat2);
		eTLS2 = tEnergy;
		if(me==0)  fprintf(screen, "UPDATE-Match N (%f, %f, %f): V1 = %f, V2 = %f \n", lDistDiff, hDistDiff, mDistDiff, tEnergy - eTLS1, tEnergy - eTLS2);
	}
	return;
}

//Calculates the difference between two minima.  Now, it finds the mass-weighted distance between vectors.
double Ridge::ComputeDistance(double** pos1, double** pos2)
{
        double dist = 0.0;
        double atomDist;
        double diff;
        double mTot = 0.0;
	double distCriteria = 0.01;
        double* m = atom->mass;
        int* type = atom->type;
        int me;
        MPI_Comm_rank(world,&me);

        for(int i=0; i<atom->nlocal;i++)
        {
                diff = 0.0;
                atomDist = 0.0;
                for(int j=0; j<domain->dimension;j++)
                {
                        diff = pos2[i][j]-pos1[i][j];
                        if(diff < -domain->prd_half[j])
                        {
                                diff = diff + domain->prd[j];
                        }
                        else if(diff > domain->prd_half[j])
                        {
                                diff = diff - domain->prd[j];
                        }
                        atomDist = atomDist + diff*diff;
                }
                atomDist = sqrt(atomDist);
		if(atomDist > distCriteria)
		{
			mTot = mTot + m[type[i]];
			dist = dist + m[type[i]]*atomDist;
		}
        }

        double commMassDist  [2]= {dist,mTot};
        double finMassDist [2];
        MPI_Allreduce(commMassDist,finMassDist,2,MPI_DOUBLE,MPI_SUM,world);
        if(finMassDist[1]<1e-6) return 0.0;
        return finMassDist[0]/finMassDist[1];
}

double Ridge::CallMinimize()
{
	int Steps = nMRelSteps;
	int maxLoops = 10;
	int me;
	char cSteps[10];
	char cFSteps[10];
	MPI_Comm_rank(world,&me);
        char **newarg = new char*[4];
        newarg[0] = (char *) "0.0";
        newarg[1] = (char *) "1.0e-6";
	ConvertIntToChar(cSteps,Steps);
        newarg[2] = cSteps;
	ConvertIntToChar(cFSteps,10*Steps);
        newarg[3] = cFSteps;
	for(int i = 0; i < maxLoops; i++)
	{
		Minimize* rMin = new Minimize(lmp);
		rMin->command(4, newarg);
		delete rMin;
		if(update->minimize->stop_condition<2)
		{
			if(me==0) fprintf(screen, "Minimization did not converge, increasing max steps to %d and max force iterations to %d.\n", Steps, Steps*10);
			Steps = Steps * 5;
			ConvertIntToChar(cFSteps,Steps);
			ConvertIntToChar(cFSteps,10*Steps);
		}
		else break;
	}
	return update->minimize->efinal;
}

void Ridge::ConvertIntToChar(char *copy, int n)
{
        std::ostringstream oss;
        oss << n;
        std::string dStr = oss.str();
        std::strcpy(copy,dStr.c_str());
	return;
}

bool Ridge::CheckSaddle(double** pos)
{
	int me;
	float eps = 1e-6;
	MPI_Comm_rank(world,&me);
	if(me==0) fprintf(screen, "Checking Saddle.\n");
        char **newarg = new char*[4];
        newarg[0] = (char *) "0.0";
        newarg[1] = (char *) "0.0";
        newarg[2] = (char *) "0";
        newarg[3] = (char *) "0";
        Minimize* rMin = new Minimize(lmp);
	CopyAtoms(atom->x,pos);
        rMin->command(4, newarg);
        delete rMin;
	if(update->minimize->fnorminf_final < epsF)
	{
		int nNeg = 0;
		int nPos = 0;
		int iSaddleCheck = InitHessianCompute();
		Compute* hessian = modify->compute[iSaddleCheck];
		hessian->compute_array();
		int ndof = 3*atom->natoms;
		for(int i = 0; i < ndof; i++)
		{
			//for(int j =0; j < ndof; j++) if(me==0) std::cout << hessian->array[i][j] << std::endl;
			if(hessian->array[i][0]>eps) nPos++;
			else if(hessian->array[i][0]<(-eps)) nNeg++;
		}
		if(nNeg == 1)
		{
			if(me==0) fprintf(screen, "UPDATE-Passes Saddle Point check.\n");
			CopyAtoms(pTLSs, pos);
			modify->delete_compute("SaddleCheck");
			return true;
		}
		if(me==0) fprintf(screen, "UPDATE-Fails Saddle Point check with %d negative entries.\n", nNeg);
		modify->delete_compute("SaddleCheck");
	}
	return false;
}

int Ridge::InitHessianCompute()
{ 
	// Create hessian compute
	char **newarg = new char*[5];
	newarg[0] = (char *) "SaddleCheck";
	newarg[1] = (char *) "all";
	newarg[2] = (char *) "freq";
	newarg[3] = (char *) "0.01";
	modify->add_compute(4,newarg);

	int iSaddleCheck = modify->find_compute("SaddleCheck");

	delete [] newarg;
	return iSaddleCheck;
}


void Ridge::InitAtomArrays()
{
        char **newarg = new char*[5];
        newarg[0] = (char *) "TLSl";
        newarg[1] = (char *) "all";
        newarg[2] = (char *) "STORE";
        newarg[3] = (char *) "0";
        newarg[4] = (char *) "3";

//Adds the Fix, and stores the pos1 array in the astore variable of the StoreFix command.
        modify->add_fix(5,newarg);
	int iTLSl = modify->find_fix((char *) "TLSl");
        FixStore *TLSl = (FixStore *) modify->fix[iTLSl];
        lAtoms = TLSl->astore;
        

//Changes the argument of the input so that the second fix created has the label 'TLS2'.        
        
        newarg[0] = (char *) "TLSh";

//Adds the Fix, and stores the pos2 array in the astore variable of the StoreFix command.
        modify->add_fix(5,newarg); 
	int iTLSh = modify->find_fix((char *) "TLSh");
        FixStore *TLSh = (FixStore *) modify->fix[iTLSh];
        hAtoms = TLSh->astore;
        
        newarg[0] = (char *) "TLSt";
        
        modify->add_fix(5,newarg);
	int iTLSt = modify->find_fix((char *) "TLSt");
        FixStore *TLSt = (FixStore *) modify->fix[iTLSt];
        tAtoms = TLSt->astore;

	//These lattice arrays are only used locally and don't need to be communicated across processors, so it isn't necessary to store them in a FixStoreLat object.  
	hLat = lLat = tLat = NULL;
	memory->grow(hLat,9,"ridge:hLat");
	memory->grow(lLat,9,"ridge:lLat");
	memory->grow(tLat,9,"ridge:tLat");
	//Initialize the values in the array to 0.0
	for(int i=0;i<9;i++)
	{
		hLat[i] = lLat[i] = tLat[i] = 0.0;
	}
	
        return;
}

void Ridge::UpdateMapping()
{

        for (int i = 0; i < atom->nlocal; i++) domain->remap(atom->x[i],atom->image[i]);
        if (domain->triclinic) domain->x2lamda(atom->nlocal);
        domain->reset_box();
        Irregular *irregular = new Irregular(lmp);
        irregular->migrate_atoms(1);
        delete irregular;
        if (domain->triclinic) domain->lamda2x(atom->nlocal);

	return;
}

void Ridge::CopyBoxToLat(double *latVector)
{
        latVector[0] = domain->boxlo[0];
        latVector[1] = domain->boxlo[1];
        latVector[2] = domain->boxlo[2];
        latVector[3] = domain->boxhi[0];
        latVector[4] = domain->boxhi[1];
        latVector[5] = domain->boxhi[2];
        latVector[6] = domain->xy;
        latVector[7] = domain->xz;
        latVector[8] = domain->yz;
        return;
}

void Ridge::CopyLatToBox(double *latVector)
{
        domain->boxlo[0] = latVector[0];
	domain->boxlo[1] = latVector[1];
        domain->boxlo[2] = latVector[2];
        domain->boxhi[0] = latVector[3];
	domain->boxhi[1] = latVector[4];
	domain->boxhi[2] = latVector[5];
	domain->xy = latVector[6];
	domain->xz = latVector[7];
	domain->yz = latVector[8];

	ResetBox();
	UpdateMapping();

        return;
}

void Ridge::CopyLatToLat(double *copyArray, double *templateArray)
{
	for(int i=0; i<9; i++)
	{
		copyArray[i] = templateArray[i];
	}
	return;
}

void Ridge::ResetBox()
{
	domain->set_initial_box();
	domain->set_global_box();
	domain->set_local_box();
}
