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
#include "output.h"
#include "memory.h"
#include "min.h"
#include "minimize.h"
#include "atom.h"
#include "fix.h"
#include "fix_store.h"
#include "ridge.h"
#include <iostream>

using namespace LAMMPS_NS;

#define MAXLINE 2048

/* ---------------------------------------------------------------------- */

Ridge::Ridge(LAMMPS *lmp) : Pointers(lmp) {}

/* ---------------------------------------------------------------------- */

void Ridge::command(int narg, char **arg){
	
	if(narg<3) error->all(FLERR,"Ridge Method -- Illegal run command");

	nRSteps = force->numeric(FLERR,arg[0]);
	nBSteps = force->numeric(FLERR,arg[1]);
	epsT = force->numeric(FLERR,arg[2]);
	nRelSteps = 10;

	PerformRidge();
	
	return;
}

void Ridge::PerformRidge()
{
        double** lAtoms = InitAtomArray();
        double** hAtoms = InitAtomArray();
        double** tAtoms = InitAtomArray();
	double lEnergy;
	double hEnergy;
	double tEnergy;
        int me;
        MPI_Comm_rank(world,&me);
	if(me==0) std::cout << "-------" << std::endl;
	if(LoadPositions(lAtoms, hAtoms)<0) error->all(FLERR,"Ridge Method -- Atomic positions not stored in fix_store.");
        if (atom->map_style == 0) {
                atom->nghost = 0;
                atom->map_init();
                atom->map_set();
        }

	PartialRelax(lAtoms, hAtoms);
	//Runs a couple tests to check that the BisectPositions function is working correctly.
	//TestBisect();	
	/*for(int i=0; i<nRSteps;i++)
	{
		for(int j=0; j<nBSteps;j++)
		{
			BisectPositions(lAtoms, hAtoms, tAtoms);

			CheckSaddle(tAtoms);
			if(sFlag)
			{
				writeTLS(lEnergy,hEnergy,tEnergy);
			}

			ComparePositions(lAtoms, hAtoms, tAtoms);
		}

		//if(sFlag) break;
		PartialRelax(lAtoms, hAtoms);
			


	}*/

	if(atom->map_style != 0)
	{
                atom->map_delete();
                atom->map_style = 0;
	}
	return;
}

void Ridge::BisectPositions(double** pos1, double** pos2, double** posOut)
{
	int m;
	double diff = 0.0;

	//Fills posOut with the values from pos1.  These should be shifted only for local atoms that move between pos1 and pos2.
	CopyAtoms(posOut,pos1);

	//Loops over atoms, using the mapping from atom->map.  If the atom is owned by the processor, its position is shifted by the difference between pos2 and pos1 for that atom.
	for(int i=1;i<=atom->natoms;i++)
	{
		m = atom->map(i);
		//if(mask[m])
		if(m>=0 && m<atom->nlocal)
		{
			for(int j=0; j<domain->dimension;j++)
			{
				diff = pos2[m][j] - pos1[m][j];
				//This if-statement ensures that the difference is not being computed across the unit cell if the atoms are just moving from one side of the unit cell to the other.
				if(diff < -domain->prd_half[j])
				{
					diff = diff + domain->prd[j];
				}
				else if(diff > domain->prd_half[j])
				{
					diff = diff - domain->prd[j];
				}
				posOut[m][j] = posOut[m][j] + 0.5*diff;
			}
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

int Ridge::LoadPositions(double** pos1, double** pos2)
{
	int me;
        int m;
        MPI_Comm_rank(world,&me);
	//First, get the labels for the fixes for the TLS atom positions.
	int iTLS1 = modify->find_fix((char *) "TLS1");
	int iTLS2 = modify->find_fix((char *) "TLS2");

	//If there are no corresponding fixes, returns -1 to flag the error.
	if((iTLS1<0)||(iTLS2<0)) return -1;

	//Creates a fix according to the stored fix
	FixStore* fix1 = (FixStore *) modify->fix[iTLS1];
	FixStore* fix2 = (FixStore *) modify->fix[iTLS2];

	//Copies the array in the FixStore to the arrays used within this class.
	CopyAtoms(pos1, fix1->astore);
	CopyAtoms(pos2, fix2->astore);

	return 0;
}

void Ridge::ReadPositions(double** pos1, double** pos2)
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
        CopyAtoms(pos1, atom->x);

	readInput[0] = (char *) "TLS2.dump";
        bisRead->command(ni, readInput);
        CopyAtoms(pos2, atom->x);
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
        fprintf(fp, "%f\t%f \n",
                         Asym, Barrier);
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
        oss << nRelSteps;
        std::string sRelSteps = oss.str();
        std::strcpy(cRelSteps,sRelSteps.c_str());

        newarg[0] = (char *) "1.0e-4";
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
