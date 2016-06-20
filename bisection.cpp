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
#include "dump.h"
#include <iostream>
#include "write_restart.h"

using namespace LAMMPS_NS;

#define MAXLINE 2048

/* ---------------------------------------------------------------------- */

Bisection::Bisection(LAMMPS *lmp) : Pointers(lmp)
{
        nMRelSteps = 1000;
	maxAlphaSteps = 15;
}

/* ---------------------------------------------------------------------- */

void Bisection::command(int narg, char **arg){
	
	if(narg<4) error->all(FLERR,"Bisection Method -- Illegal run command");

	bigint nsteps_input = force->bnumeric(FLERR,arg[0]);
	epsT = force->numeric(FLERR,arg[1]);
	inputSetFlag = 0;
	


	int inflag = 0;

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
	//double epsT = 0.02;
	bigint intCurrStep = 0;
	bigint maxSteps = log2(nsteps)+5;
	bigint lowerStep=0, higherStep=nsteps;
	char *charCurrStep = new char[50];
	int minIndex1;
	int minIndex2;
	double lEnergyMin;
	double hEnergyMin;
	double tEnergyMin;
	double eDiff;
	double lDistDiff;
	double hDistDiff;
        int me;
        MPI_Comm_rank(world,&me);

	if(me==0) OpenTLS();

	if(me==0) fprintf(logfile, "epsT is %f\n",epsT);

	//Need to prepare input commands for the read_dump command.  The arguments to the command need to be organized in
	//char **, which contains the string "[filename] [step] x y [z] replace yes".  That string is created here,
	//and then passed to the ConvertToChar function, which creates parses the string and converts it to char **.
	//Later, compare to something like what is in compute_msd.cpp for a better implementation.
	char** readInput;
	readInput = NULL;
	readInput = (char **) memory->srealloc(readInput,6*sizeof(char *),"bisection:RDargs");
	std::string strInput = bisFilename;
	if(domain->dimension==2) {
		strInput = strInput + " 0 x y replace yes";
	}
	else{
		strInput = strInput + " 0 x y z replace yes";
	}

        if (atom->map_style == 0) {
                if(me==0) fprintf(screen, "Bisection: Getting new map\n");
                atom->nghost = 0;
                atom->map_init();
                atom->map_set();
        }

	InitAtomArrays();
	int nInput = ConvertToChar(readInput, strInput);

	//Creates a ReadDump class, and then has it read the appropriate timestep, using the parsed input string.
	
	ReadDump *bisRead = new ReadDump(lmp);
	
	bisRead->command(nInput, readInput);
	//Minimizes the atomic configuration and then stores the energy in lEnergyMin.
	
	lEnergyMin = CallMinimize();

	CopyAtoms(lAtoms,atom->x);
	readInput[1] = &charCurrStep[0];
	intCurrStep = UpdateDumpArgs(nsteps, charCurrStep);
	bisRead->command(nInput, readInput);

	hEnergyMin = CallMinimize();
	CopyAtoms(hAtoms,atom->x);
	
	//Test Functions
	//TestComputeDistance();
	//TestMinimize(nsteps, bisRead, nInput, readInput);
//GTH
	if(ComputeDistance(lAtoms,hAtoms)<epsT)
	{
		if(me==0) fprintf(fp,"End-points for bisection have same minimum.  Bisection may fail.\n");
	}

	bigint iSteps = 0;
	
	while(iSteps<maxSteps)
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
			if(me==0)  fprintf(screen, "UPDATE-Match L (%f, %f, %f): lower=" BIGINT_FORMAT ", higher=" BIGINT_FORMAT "\n", lEnergyMin,
                                        hEnergyMin, lDistDiff, intCurrStep, higherStep);
			lowerStep = intCurrStep;
			CopyAtoms(lAtoms, tAtoms);
			lEnergyMin = tEnergyMin;
		}
		else{
			if(hDistDiff<epsT)
			{
				if(me==0)  fprintf(screen, "UPDATE-Match U (%f, %f, %f): lower=" BIGINT_FORMAT ", higher=" BIGINT_FORMAT "\n", lEnergyMin,
                                        hEnergyMin, hDistDiff, lowerStep, intCurrStep);
				higherStep = intCurrStep;
				CopyAtoms(hAtoms, tAtoms);
				hEnergyMin = tEnergyMin;
			}
			else{
				if(me==0)  fprintf(screen, "UPDATE-Match N (%f, %f, %f): lower=" BIGINT_FORMAT ", higher=" BIGINT_FORMAT "\n", lEnergyMin,
					hEnergyMin, hDistDiff, lowerStep, intCurrStep);
				higherStep = intCurrStep;
				CopyAtoms(hAtoms, tAtoms);
				hEnergyMin = tEnergyMin;
			}
		}
		eDiff = fabs(hEnergyMin-lEnergyMin);
		iSteps++;
		if(higherStep-lowerStep<=1) break;
	}

	WriteTLS(intCurrStep,lAtoms,hAtoms,lEnergyMin,hEnergyMin);
	//StoreAtoms(lAtoms, hAtoms);
	//WriteAtoms(lAtoms, hAtoms);

	//Deletes readInput and atom arrays, to prevent memory leaks.
	
	delete bisRead;
	memory->sfree(readInput);
	//DeleteAtomArray(lAtoms);
	//DeleteAtomArray(hAtoms);
	//DeleteAtomArray(tAtoms);	
	if(me==0) fclose(fp);
	modify->delete_fix((char *) "TLSt");

    	/*if (atom->map_style!=0) {
      		atom->map_delete();
	      	atom->map_style = 0;
    	}*/
	
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


void Bisection::TestMinimize(bigint nsteps, ReadDump *bisRead, int nInput, char **readInput)
{
	double tEmin;
	bigint dummy;
        int me;
        MPI_Comm_rank(world,&me);
	for(bigint i=0;i<=nsteps;i++)
	{
		dummy = UpdateDumpArgs(i,readInput[1]);
		bisRead->command(nInput, readInput);
		tEmin = CallMinimize();
		if(me==0)  fprintf(fp, BIGINT_FORMAT "\t%f \n",
			i, tEmin);
	}
}

int Bisection::UpdateDumpArgs(bigint currStep, char *charCurrStep)
{
	std::ostringstream oss;
	oss << (long long)currStep;
	std::string strCurrStep = oss.str();
	std::strcpy(charCurrStep,strCurrStep.c_str());
	return currStep;
}

//Calculates the difference between two minima.  Now, it finds the rms difference between vectors.
//**Can definitely be optimized by using BLAS libraries to find vector differences
//and then taking the dot product between the vectors.
double Bisection::ComputeDistance(double** pos1, double** pos2)
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

void Bisection::TestComputeDistance()
{
        double** Atoms1 = lAtoms;
        double** Atoms2 = hAtoms;
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
		if(me==0) fprintf(screen, "ComputeDistance fails test 1.  Expected Diff==%f, but got %f.\n",sqrt(3), Diff);
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

	Atoms2[0][0] = 1;
        Diff = ComputeDistance(Atoms1, Atoms2);
        if(fabs(sqrt(m[type[0]]/MassTot)-Diff)<1E-3)
        {
                if(me==0) fprintf(screen, "ComputeDistance passes test 2.\n");
        }
        else
        {
                if(me==0) fprintf(screen, "ComputeDistance fails test 2.  Expected Diff==%f, but got %f.\n", sqrt(m[type[0]]/MassTot), Diff);
        }

	//Third test sets the positions to be at the edges of the unit cell.  This should give 0 displacement if the edges are treated correctly.
	for(int j=0;j<domain->dimension;j++)
	{
		Atoms1[0][j] = -(domain->prd_half[j]);
		Atoms2[0][j] = domain->prd_half[j];
	}

        Diff = ComputeDistance(Atoms1, Atoms2);
        if(fabs(Diff)<1E-3)
        {
                if(me==0) fprintf(screen, "ComputeDistance passes test 3.\n");
        }
        else
        {
                if(me==0) fprintf(screen, "ComputeDistance fails test 3.  Expected Diff==%f, but got %f.\n", 0.0, Diff);
        }

	//DeleteAtomArray(Atoms1);
	//DeleteAtomArray(Atoms2);
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
        FixStore *TLS1 = (FixStore *) modify->fix[modify->nfix-1];
        lAtoms = TLS1->astore;

//Changes the argument of the input so that the second fix created has the label 'TLS2'.        

        newarg[0] = (char *) "TLS2";

//Adds the Fix, and stores the pos2 array in the astore variable of the StoreFix command.
        modify->add_fix(5,newarg);
        FixStore *TLS2 = (FixStore *) modify->fix[modify->nfix-1];
	hAtoms = TLS2->astore;

	newarg[0] = (char *) "TLSt";

	modify->add_fix(5,newarg);
	FixStore *tTLS = (FixStore *) modify->fix[modify->nfix-1];
	tAtoms = tTLS->astore;
	return;
}

void Bisection::DeleteAtomArray(double** atomArray)
{
	memory->sfree(atomArray[0]);
	memory->sfree(atomArray);
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

//Stores atomic position arrays in a StoreFix.  StoreFix's are labelled using a char array identifier as well as indexed.  To recall a StoreFix, you can search for the index using the char array label
//as illustrated in the ridge class.  This function stores the atomic positions for both minima in StoreFix's, one labelled 'TLS1' and the other labelled 'TLS2'.
void Bisection::StoreAtoms(double** pos1, double** pos2)
{
	char **newarg = new char*[5];
	double diff = 0.0;
        int me;
        MPI_Comm_rank(world,&me);

//Created the arguments for the StoreFix
	newarg[0] = (char *) "TLS1";
	newarg[1] = (char *) "all";
	newarg[2] = (char *) "STORE";
	newarg[3] = (char *) "0";
	newarg[4] = (char *) "3";

//Adds the Fix, and stores the pos1 array in the astore variable of the StoreFix command.
	modify->add_fix(5,newarg);
	FixStore *TLS1 = (FixStore *) modify->fix[modify->nfix-1];
	if(me==0) std::cout << "Storing Atoms" << std::endl;
	//std::cout << me << "    " << atom->nlocal << std::endl << std::flush;
	CopyAtoms(TLS1->astore,pos1);
	
	/*for(int i = 0; i < atom->nlocal; i++)
	{
		std::cout << me << ", " << i << ", " << pos2[i][0] << ", " << pos2[i][1] << ", " << pos2[i][2] << std::endl << std::flush;
	}*/

//Changes the argument of the input so that the second fix created has the label 'TLS2'.	
/*
	newarg[0] = (char *) "TLS2";

//Adds the Fix, and stores the pos2 array in the astore variable of the StoreFix command.
	modify->add_fix(5,newarg);
        FixStore *TLS2 = (FixStore *) modify->fix[modify->nfix-1];
	CopyAtoms(TLS2->astore,pos2);
*/
	MPI_Barrier(world);
	delete [] newarg;
	return;
}

void Bisection::WriteAtoms(double** pos1, double** pos2)
{
        char **newarg = new char*[1];
        int me;
        MPI_Comm_rank(world,&me);

        newarg[0] = (char *) "TLS_Scratch/restart1.%";
        CopyAtoms(atom->x,pos1);
	WriteRestart *TLS1 = new WriteRestart(lmp);
	TLS1->command(1,newarg);
	delete TLS1;

        newarg[0] = (char *) "TLS_Scratch/restart2.%";
        CopyAtoms(atom->x,pos2);
	WriteRestart *TLS2 = new WriteRestart(lmp);
	TLS2->command(1,newarg);
	delete TLS2;

        delete [] newarg;
	return;
}
