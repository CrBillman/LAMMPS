/*
 * bisection.cpp
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
#include "fix_store_lat.h"
#include "dump.h"
#include <iostream>
#include "write_restart.h"

using namespace LAMMPS_NS;

#define MAXLINE 2048

/* ---------------------------------------------------------------------- */

Bisection::Bisection(LAMMPS *lmp) : Pointers(lmp)
{
        nMRelSteps = 1000;     				//The number of relaxation steps to start with for CallMinimize()
	maxAlphaSteps = 5;    				//The maximum number of addition minimizations to do because of the 'alpha linsearch' stopping condition.  For larger values, it helps the minimization actually move toward the true minimum.
        matchExit = true;                               //Determines whether or not to exit if the end-points of the trajectory relax to the same minimum
}

/* ---------------------------------------------------------------------- */

void Bisection::command(int narg, char **arg)
{
	//Gets the arguments from the input line	
	if(narg<4) error->all(FLERR,"Bisection Method -- Illegal run command");
	bigint nsteps_input = force->bnumeric(FLERR,arg[0]);	//The number of steps in the input trajectory
	epsT = force->numeric(FLERR,arg[1]);			//Tolerance for the distance criteria for defining different minima

	inputSetFlag = 0;					
	int iarg=2;
	while(iarg<narg){
		if(strcmp(arg[iarg],"FMD") == 0){
		    if (iarg+1 > narg) error->all(FLERR,"Illegal run command");
			char* bisFilename = arg[iarg+1];
			BisectionFromMD(nsteps_input, bisFilename);
			inputSetFlag = 1;
			iarg++;
		}
		iarg++;
	}

	if(inputSetFlag==0) error->all(FLERR,"Bisection Method -- No input method selected");
	
	return;

}

void Bisection::BisectionFromMD(bigint nsteps, char* bisFilename){

	//Initialize all bisection variables
	bigint intCurrStep = 0;
	bigint maxSteps = log2(nsteps)+5;
	bigint lowerStep=0, higherStep=nsteps;
	char *charCurrStep = new char[50];
	char **readInput;
	int nInput;
	int me;
	int minIndex1;
	int minIndex2;
	double lEnergyMin;
	double hEnergyMin;
	double tEnergyMin;
	double eDiff;
	double lDistDiff;
	double hDistDiff;
        MPI_Comm_rank(world,&me);

	//Opens TLS.dump, where the output of the TLS search is written
	if(me==0) OpenTLS();

	//Prepares input commands for the read_dump command.  If-statement handles 2d simulations
	if(domain->dimension==2)
	{
		readInput = new char*[6];
		readInput[0] = (char *) bisFilename;
		readInput[1] = (char *) "0";
		readInput[2] = (char *) "x";
		readInput[3] = (char *) "y";
		readInput[4] = (char *) "replace";
		readInput[5] = (char *) "yes";
		nInput = 6;
	}
	else
	{
                readInput = new char*[7];
                readInput[0] = (char *) bisFilename;
                readInput[1] = (char *) "0";
                readInput[2] = (char *) "x";
                readInput[3] = (char *) "y";
		readInput[4] = (char *) "z";
                readInput[5] = (char *) "replace";
                readInput[6] = (char *) "yes";
		nInput = 7;
	}

	//Get a mapping, to handle per-atom arrays.
        if (atom->map_style == 0) {
                if(me==0) fprintf(screen, "Bisection: Getting new map\n");
                atom->nghost = 0;
                atom->map_init();
                atom->map_set();
        }

	//Initialize arrays for storing atomic positions and lattice vectors.  These are stored in FixStore and FixStoreLat objects, respectively.  This handles the migration of atoms across processors inherently
	InitAtomArrays();

	//Creates a ReadDump class, and then has it read the appropriate timestep, using the parsed readInput.
	ReadDump *bisRead = new ReadDump(lmp);
	bisRead->command(nInput, readInput);

	//Minimizes the first atomic configuration from the dump file and then stores the energy in lEnergyMin.
	lEnergyMin = CallMinimize();
	CopyAtoms(lAtoms,atom->x);
	CopyBoxToLat(lat1);

        //Updates readInput to go to point toward the current step, and updates the current step to be the last step in the trajectory.  
        //Then, loads and minimizes that atomic configuration from the dump file and then stores the energy in hEnergyMin.	
	readInput[1] = &charCurrStep[0];
	intCurrStep = UpdateDumpArgs(nsteps, charCurrStep);
	bisRead->command(nInput, readInput);
	hEnergyMin = CallMinimize();
	CopyAtoms(hAtoms,atom->x);
	CopyBoxToLat(lat2);

	//Checks the mass-weighted distance between the minimized configurations of the first and last timesteps of the trajectory.  If they are in the same minimum (ie, the mw distance is smaller than the cut-off), then it is unlikely that a TLS
	//will be found in the trajectory.  If matchExit is true, the bisection method is exited at this point.  If false, it only outputs a warning.	
	if(ComputeDistance(lAtoms,hAtoms)<epsT)
	{
		if(matchExit) 
		{
			if(me==0) fprintf(screen,"UPDATE-Exiting bisection, as end-points for bisection have same minimum.\n");
			return;
		}
		if(me==0) fprintf(screen,"UPDATE-End-points for bisection have same minimum.  Bisection may fail.\n");
	}

	while(true)
	{
		intCurrStep = UpdateDumpArgs((higherStep-lowerStep)/2+lowerStep,charCurrStep);
		bisRead->command(nInput, readInput);
		CopyAtoms(tAtoms,atom->x);
		tEnergyMin = CallMinimize();
		eDiff = fabs(tEnergyMin-lEnergyMin);
		lDistDiff = ComputeDistance(lAtoms,atom->x); 
		hDistDiff = ComputeDistance(hAtoms,atom->x);
		if((lDistDiff<hDistDiff)&&(lDistDiff<epsT))
		{
			lowerStep = intCurrStep;
			CopyAtoms(lAtoms, atom->x);
			CopyBoxToLat(lat1);
			lEnergyMin = tEnergyMin;
			if(me==0)  fprintf(screen, "UPDATE-Match L (%f, %f, %f): lower=" BIGINT_FORMAT ", higher=" BIGINT_FORMAT "\n", lEnergyMin,
                                        hEnergyMin, lDistDiff, intCurrStep, higherStep);
		}
		else{
			if(hDistDiff<epsT)
			{
				higherStep = intCurrStep;
				CopyAtoms(hAtoms, atom->x);
				CopyBoxToLat(lat2);
				hEnergyMin = tEnergyMin;
				if(me==0)  fprintf(screen, "UPDATE-Match U (%f, %f, %f): lower=" BIGINT_FORMAT ", higher=" BIGINT_FORMAT "\n", lEnergyMin,
                                        hEnergyMin, hDistDiff, lowerStep, intCurrStep);
			}
			else{
				higherStep = intCurrStep;
				CopyAtoms(hAtoms, atom->x);
				CopyBoxToLat(lat2);
				hEnergyMin = tEnergyMin;
				if(me==0)  fprintf(screen, "UPDATE-Match N (%f, %f, %f, %f): lower=" BIGINT_FORMAT ", higher=" BIGINT_FORMAT "\n", lEnergyMin,
                                        hEnergyMin, lDistDiff, hDistDiff, lowerStep, intCurrStep);
			}
		}
		if(higherStep-lowerStep<=1) break;
	}

	eDiff = fabs(hEnergyMin-lEnergyMin);
	WriteTLS(intCurrStep,lAtoms,hAtoms,lEnergyMin,hEnergyMin);

	//Deletes readInput and atom arrays, to prevent memory leaks.
	
	delete bisRead;
	delete readInput;
	if(me==0) fclose(fp);
	modify->delete_fix((char *) "TLSt");

	return;
}

//Converts a std::string into a char **, which is required for interfacing with several of the classes and functions
//within LAMMPS.
int Bisection::ConvertToChar(char ** charArray, std::string strInput)
{
	int nArgs = 0;
	char *charInput = new char[strInput.length()+1];
	std::strcpy(charInput,strInput.c_str());
	char *start = &charInput[0];
	char *stop;
	charArray[0] = start;
	while(1){
		nArgs++;
		stop = &start[strcspn(start," ")];
		if(*stop=='\0') break;
		*stop = '\0';
		start = stop+1;
		charArray[nArgs] = start;
	}

	return nArgs;
}

double Bisection::CallMinimize()
{       
        int Steps = nMRelSteps;
        int maxLoops = 40;
        int me;
	int alphaCounter = 0;
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
		else if((update->minimize->stop_condition==5)&&(alphaCounter<maxAlphaSteps))
		{
                        if(me==0) fprintf(screen, "Minimization did not converge, resubmitting to handle alpha linesearch stopping condition. Counter:%d; Max:%d.\n",alphaCounter,maxAlphaSteps);
			alphaCounter++;
		}
		else break;
	}
	return update->minimize->efinal;
}

int Bisection::UpdateDumpArgs(bigint currStep, char *charCurrStep)
{
	std::ostringstream oss;
	oss << (long long)currStep;
	std::string strCurrStep = oss.str();
	std::strcpy(charCurrStep,strCurrStep.c_str());
	return currStep;
}

//Calculates the difference between two minima.  Now, it finds the mass-weighted distance for atoms above the distCriteria.
double Bisection::ComputeDistance(double** pos1, double** pos2)
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

void Bisection::OpenTLS()
{
	std::string strFile = "TLS.dump";
	char *charFile = new char[20];
	std::strcpy(charFile,strFile.c_str());
	fp = fopen(charFile,"a");
	return;
}

void Bisection::WriteTLS(bigint step, double** x1, double** x2, double E1, double E2)
{
	double dist = ComputeDistance(x1,x2);
	double Ediff = E2 - E1;
	int me;
	MPI_Comm_rank(world, &me);
	if(me==0) fprintf(fp, "Bisection: "BIGINT_FORMAT "\t%f\t%f \n", step, Ediff, dist);
	return;
}

void Bisection::InitAtomArrays()
{
	char **newarg = new char*[5];
        newarg[0] = (char *) "TLS1";
        newarg[1] = (char *) "all";
        newarg[2] = (char *) "STORE";
        newarg[3] = (char *) "0";
        newarg[4] = (char *) "3";

//Adds the Fix, and stores the pos1 array in the astore variable of the StoreFix command.
        modify->add_fix(5,newarg);
	int iTLS1 = modify->find_fix((char *) "TLS1");
        FixStore *TLS1 = (FixStore *) modify->fix[iTLS1];
        lAtoms = TLS1->astore;

//Changes the argument of the input so that the second fix created has the label 'TLS2'.        

        newarg[0] = (char *) "TLS2";

//Adds the Fix, and stores the pos2 array in the astore variable of the StoreFix command.
        modify->add_fix(5,newarg);
	int iTLS2 = modify->find_fix((char *) "TLS2");
        FixStore *TLS2 = (FixStore *) modify->fix[iTLS2];
	hAtoms = TLS2->astore;

	newarg[0] = (char *) "TLSt";

	modify->add_fix(5,newarg);
	int iTLSt = modify->find_fix((char *) "TLSt");
	FixStore *tTLS = (FixStore *) modify->fix[iTLSt];
	tAtoms = tTLS->astore;

//Adds fixes for lattice storage.
	newarg[0] = (char *) "TLSLat1";
	newarg[2] = (char *) "STORELAT";
	modify->add_fix(3,newarg);
	int iTLSl1 = modify->find_fix((char *) "TLSLat1");
	FixStoreLat *TLSl1 = (FixStoreLat *) modify->fix[iTLSl1];
	lat1 = TLSl1->vstore;

        newarg[0] = (char *) "TLSLat2";
        modify->add_fix(3,newarg);
        int iTLSl2 = modify->find_fix((char *) "TLSLat2");
        FixStoreLat *TLSl2 = (FixStoreLat *) modify->fix[iTLSl2];
        lat2 = TLSl2->vstore;
	return;
}

void Bisection::CopyAtoms(double** copyArray, double** templateArray)
{
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

void Bisection::ConvertIntToChar(char *copy, int n)
{       
        std::ostringstream oss;
        oss << n;
        std::string dStr = oss.str();
        std::strcpy(copy,dStr.c_str());
        return;
}

void Bisection::CopyBoxToLat(double *latVector)
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
