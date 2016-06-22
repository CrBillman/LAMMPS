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
#include "ridge.h"
#include "irregular.h"
#include <iostream>
#include "compute.h"
#include "compute_freq.h"

using namespace LAMMPS_NS;

#define MAXLINE 2048

/* ---------------------------------------------------------------------- */

Ridge::Ridge(LAMMPS *lmp) : Pointers(lmp) {
	epsF = 25e-3;
	nPRelSteps = 5;
	nMRelSteps = 1000;
}


/* ---------------------------------------------------------------------- */

void Ridge::command(int narg, char **arg){
	
	if(narg<3) error->all(FLERR,"Ridge Method -- Illegal run command");

	nRSteps = force->numeric(FLERR,arg[0]);
	nBSteps = force->numeric(FLERR,arg[1]);
	epsT = force->numeric(FLERR,arg[2]);

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
	eTLS1 = CallMinimize();
	CopyAtoms(lAtoms,atom->x);
	
	CopyAtoms(atom->x,pTLS2);
	eTLS2 = CallMinimize();
	CopyAtoms(hAtoms,atom->x);
	float chDist = ComputeDistance(lAtoms,hAtoms);
	if(chDist<epsT)
	{
		if(me==0) fprintf(screen, "UPDATE-End-points relaxed to same minimum (distance is %f), leaving ridge method.\n", chDist);
		return;
	}
	if(me==0) fprintf(screen, "UPDATE-Asymmetry: %f\n",eTLS2-eTLS1);

	//TestComputeDistance();
	//Runs a couple tests to check that the BisectPositions function is working correctly.
	//TestBisect();	
	for(int i=0; i<nRSteps;i++)
	{
		for(int j=0; j<nBSteps;j++)
		{
			BisectPositions(lAtoms, hAtoms, tAtoms);
			sFlag = CheckSaddle(tAtoms);
			//if(me==0) fprintf(screen, "Ridge step: %d.  Bisection Step: %d. x-pos: %f. Force norm: %f\n",i,j,atom->x[1][0],update->minimize->fnorm2_init);
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

	if(!sFlag) if(me==0) fprintf(screen, "Cannot find saddle.\n");

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
			//std::cout << i << ", " << j << ": " << hAtoms[i][j] << ", " << lAtoms[i][j] << std::endl;
		}
	}

	return;
}

void Ridge::TestBisect()
{
	double** ar1 = InitAtomArray();
	double** ar2 = InitAtomArray();
	double** ar3 = InitAtomArray();
	double tol = 1E-6;
        int me;
	int m;
	bool failFlag = false;
        MPI_Comm_rank(world,&me);

	//Simple test, should only fail if arrays are incorrectly built, or if something is accessing memory where it shouldn't.  This test checks if the bisected between 0 is 0.
	for(int i=0;i<atom->natoms;i++)
	{
		for(int j=0;j<domain->dimension;j++)
		{
			ar1[i][j] = 0.0;
			ar2[i][j] = 0.0;
		}
	}

	BisectPositions(ar1, ar2, ar3);
	for(int i=1;i<=atom->natoms;i++)
	{
		for(int j=0;j<domain->dimension;j++)
		{
			m = atom->map(i);
			//if (mask[m]) {
			if(m>=0 && m<atom->nlocal)
			{
				if(ar3[m][0]>tol)
				{
					if(me==0) fprintf(logfile,"BisectPosition fails first test, on value %d,%d\n",m,j);
					failFlag = true;
					break;
				}
			}
		}
		if(failFlag) break;
	}
	if(me==0 && !failFlag) fprintf(logfile,"BisectPosition passed first test.\n");

	//This test checks for a simple bisection.  It sets the position of the first atom in ar2 array to 1.0 in each direction.  As the position in ar1 is 0.0, it should return 0.5 in each 
	//direction for that element.
	failFlag = false;
	for(int j=0;j<domain->dimension;j++)
	{
		ar2[0][j] = 1.0;
	}

	BisectPositions(ar1, ar2, ar3);	
	for(int j=0;j<domain->dimension;j++)
	{
		if((ar3[0][j]-0.5)>tol)
		{
			if(me==0) fprintf(logfile,"BisectPosition fails second test, in dimension %d, where value is %f\n",j,ar3[0][j]);
			failFlag = true;
		}
		break;
	}
	if(me==0 && !failFlag) fprintf(logfile,"BisectPosition passed second test.\n");

	//This checks for the literal edge case where atoms are placed at the outside edges of the unit cell.  A naive bisection would put the atom here in the center of the unit cell.
	//However, their true difference is 0, so a bisection shouldn't shift them.
	for(int i=1;i<=atom->natoms;i++)
	{
		m = atom->map(i);
		//if (mask[m]) {
		if(m>=0 && m<atom->nlocal)
		{
			for(int j=0;j<domain->dimension;j++)
			{       
				ar1[m][j] = -(domain->prd_half[j]);
				ar2[m][j] = domain->prd_half[j];
			}
		}
        }   

	BisectPositions(ar1, ar2, ar3);
	
	failFlag = false;
	for(int i=1;i<=atom->natoms;i++)
        {
		m = atom->map(i);
		//if (mask[m]) {
		if(m>=0 && m<atom->nlocal)
		{
			for(int j=0;j<domain->dimension;j++)
			{
				if((ar3[m][j]-domain->prd_half[j])>tol)
				{
					if(me==0) fprintf(logfile,"BisectPosition fails third test, at index %d,%d with value %f\n",m,j,ar3[m][j]);
					failFlag = true;
					break;
				}
			}
		}
		if(failFlag) break;
        }
	if(me==0 && !failFlag) fprintf(logfile,"BisectPosition passed third test.\n");

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

	//If there are no corresponding fixes, returns -1 to flag the error.
	if((iTLS1<0)||(iTLS2<0)) return -1;

	//Creates a fix according to the stored fix
	FixStore* TLS1 = (FixStore *) modify->fix[iTLS1];
	FixStore* TLS2 = (FixStore *) modify->fix[iTLS2];

        if(me==0) std::cout << "Loading Atoms" << std::endl;

	//Copies the array in the FixStore to the arrays used within this class.
	//CopyAtoms(atom->x,TLS1->astore);
	pTLS1 = TLS1->astore;
	pTLS2 = TLS2->astore;


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
        double Asym = E2 - E1;
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
	pDump->command(8,dumparg);
	update->reset_timestep(1);
	CopyAtoms(atom->x,pTLS2);
	pDump->command(8,dumparg);
	update->reset_timestep(2);
        CopyAtoms(atom->x,pTLSs);
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

	if(me==0) fprintf(screen,"Performing first ridge min, with newarg = %s\n",newarg[2]);

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
	CallMinimize();
	tEnergy = update->minimize->einitial;

	lDistDiff = ComputeDistance(atom->x, pTLS1);
	hDistDiff = ComputeDistance(atom->x, pTLS2);
	mDistDiff = ComputeDistance(pTLS1, pTLS2);
	if((lDistDiff<epsT) && (lDistDiff<hDistDiff))
	{
		CopyAtoms(lAtoms,tAtoms);
		if(me==0)  fprintf(screen, "UPDATE-Match L (%f, %f, %f): V1 = %f, V2 = %f \n", lDistDiff, hDistDiff, mDistDiff, tEnergy - eTLS1, tEnergy - eTLS2);
	}
	else if(hDistDiff<epsT)
	{
		CopyAtoms(hAtoms,tAtoms);
		if(me==0)  fprintf(screen, "UPDATE-Match U (%f, %f, %f): V1 = %f, V2 = %f \n", lDistDiff, hDistDiff, mDistDiff, tEnergy - eTLS1, tEnergy - eTLS2);
	}
	else
	{
		CopyAtoms(hAtoms,tAtoms);
		CopyAtoms(pTLS2, atom->x);
		eTLS2 = tEnergy;
		if(me==0)  fprintf(screen, "UPDATE-Match N (%f, %f, %f): V1 = %f, V2 = %f \n", lDistDiff, hDistDiff, mDistDiff, tEnergy - eTLS1, tEnergy - eTLS2);
	}
	return;
}

double Ridge::ComputeDistance(double** pos1, double** pos2)
{
	double dist = 0.0;
	double diff;
	double mTot = 0.0;
	double* m = atom->mass;
	int* type = atom->type;
	int me;
	MPI_Comm_rank(world,&me);
	for(int i=0; i<atom->nlocal;i++)
	{
		diff = 0.0;
		mTot = mTot + m[type[i]];
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
			dist = dist + m[type[i]]*diff*diff;
		}
	}

	double commMassDist  [2]= {dist,mTot};
	double finMassDist [2];

	MPI_Allreduce(commMassDist,finMassDist,2,MPI_DOUBLE,MPI_SUM,world);

	return sqrt(finMassDist[0]/finMassDist[1]);
}
void Ridge::TestComputeDistance()
{
        double** Atoms1 = InitAtomArray();
        double** Atoms2 = InitAtomArray();
	double Diff = 0.0;
	double* m = atom->mass;
	int* type = atom->type;
	double MassTot = 0.0;
	int me;
        MPI_Comm_rank(world,&me);

	//First test sets the position of every atom in Atoms1 to {0,0,0} and Atoms2 to {1,1,1}.  If the difference is being calculated correctly, Diff should be sqrt(3).
	for(int i=0;i<atom->natoms;i++)
	{
		for(int j=0;j<domain->dimension;j++)
		{
			Atoms1[i][j] = 0.0;
			Atoms2[i][j] = 1.0;
		}
	}

	Diff = ComputeDistance(Atoms1, Atoms2);
	if(fabs(sqrt(3)-Diff)<1E-3)
	{
		if(me==0) fprintf(screen, "ComputeDistance passes test 1.\n");
	}
	else
	{
		if(me==0) fprintf(screen, "Computedifference fails test 1.  Expected Diff==%f, but got %f.\n",sqrt(3), Diff);
	}
	
	//Second test sets the positions in both Atoms arrays to {0,0,0}, except a single entry in Atoms2, which is set to {1,0,0}.  This is to test the mass weighting.
        for(int i=0;i<atom->natoms;i++)
        {
		MassTot = MassTot + m[type[i]];
                for(int j=0;j<domain->dimension;j++)
                {
                        Atoms2[i][j] = 0.0;
                }
        }

	if(me==0) Atoms2[0][0] = 1;
        Diff = ComputeDistance(Atoms1, Atoms2);
        if(fabs(sqrt(m[type[0]]/MassTot)-Diff)<1E-3)
        {
                if(me==0) fprintf(screen, "ComputeDistance passes test 2.\n");
        }
        else
        {
                if(me==0) fprintf(screen, "Computedifference fails test 2.  Expected Diff==%f, but got %f.\n", sqrt(m[type[0]]/MassTot), Diff);
        }

	//Third test sets the positions to be at the edges of the unit cell.  This should give 0 displacement if the edges are treated correctly.
	if(me==0)
	{
		for(int j=0;j<domain->dimension;j++)
		{
			Atoms1[0][j] = -(domain->prd_half[j]);
			Atoms2[0][j] = domain->prd_half[j];
		}
	}

        Diff = ComputeDistance(Atoms1, Atoms2);
        if(fabs(Diff)<1E-3)
        {
                if(me==0) fprintf(screen, "ComputeDistance passes test 3.\n");
        }
        else
        {
                if(me==0) fprintf(screen, "Computedifference fails test 3.  Expected Diff==%f, but got %f.\n", 0.0, Diff);
        }

	DeleteAtomArray(Atoms1);
	DeleteAtomArray(Atoms2);
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
