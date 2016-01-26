/*
 * bisection.cpp
 *
 *  Created on: Jan 11, 2016
 *      Author: Chris
 */

#include <stdlib.h>
#include <string.h>
#include <math.h>
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
#include "atom.h"

using namespace LAMMPS_NS;

#define MAXLINE 2048

/* ---------------------------------------------------------------------- */

Bisection::Bisection(LAMMPS *lmp) : Pointers(lmp) {}

/* ---------------------------------------------------------------------- */

void Bisection::command(int narg, char **arg){
	//fprintf(screen, "I'm here!");
	if(narg<3) error->all(FLERR,"Bisection Method -- Illegal run command");

	bigint nsteps_input = force->bnumeric(FLERR,arg[0]);
	inputSetFlag = 0;
	//fprintf(screen, "nsteps_input is " BIGINT_FORMAT, nsteps_input);
	//Sets up minimization arguments.
	/*
	update->etol = force->numeric(FLERR,arg[1]);
	update->ftol = force->numeric(FLERR,arg[2]);
	update->nsteps = force->inumeric(FLERR,arg[3]);
	update->max_eval = force->inumeric(FLERR,arg[4]);
	*/
	InitializeMinimize();


	int inflag = 0;

	int iarg=1;
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
	double epsE = 0.01;
	double epsT = 0.01;
	bigint intCurrStep = 0;
	bigint lowerStep=0, higherStep=nsteps;
	char *charCurrStep = new char[50];
	int minIndex1;
	int minIndex2;
	double minEnergy1;
	double minEnergy2;
	double minEnergy3;
	double** atoms1;
	double** atoms2;
	double** tAtoms;

	//Need to prepare input commands for the read_dump command.  The arguments to the command need to be organized in
	//char **, which contains the string "[filename] [step] x y [z] replace yes".  That string is created here,
	//and then passed to the ConvertToChar function, which creates parses the string and converts it to char **.
	char** readInput;
	readInput = NULL;
	readInput = (char **) memory->srealloc(readInput,5*sizeof(char *),"bisection:RDargs");
	std::string strInput = bisFilename;
	if(domain->dimension==2) {
		strInput = strInput + " 0 x y replace yes";
	}
	else{
		strInput = strInput + " 0 x y z replace yes";
	}
	int nInput = ConvertToChar(readInput, strInput);

	//Creates a ReadDump class, and then has it read the appropriate timestep, using the parsed input string.
	ReadDump* bisRead = new ReadDump(lmp);
	bisRead->command(nInput, readInput);

	//Minimizes the atomic configuration and then stores the energy in minEnergy1.
	minEnergy1 = CallMinimize();
	atoms1 = atom->x;
	readInput[1] = &charCurrStep[0];
	intCurrStep = UpdateDumpArgs(nsteps, charCurrStep);
	bisRead->command(nInput, readInput);
	minEnergy2 = CallMinimize();
	atoms2 = atom->x;

	if(((minEnergy2-minEnergy1)<epsE)&&(ComputeDifference(atoms1,atoms2)<epsT))
	{
		fprintf(screen,"End-points for bisection have same minimum.  Bisection may fail.");
	}
	//fprintf(screen,"The readInput[1] value is %s and the x position of the first atom is %f \n", readInput[1],atoms1[0][0]);

	//Creates a Minimize class, and then has it minimize the loaded atomic configuration, using the parsed arguments.

	bigint iSteps = 0;
	int extraMin = 0;
	bigint* extraList;
	extraList = (bigint *) memory->smalloc(10*sizeof(bigint),
            "bisection:extraList");

	OpenTLS();

	while(iSteps<(log2(nsteps)+1))
	{
		intCurrStep = UpdateDumpArgs((higherStep-lowerStep)/2+lowerStep,charCurrStep);
		minEnergy3 = CallMinimize();
		tAtoms = atom->x;
		if((minEnergy3-minEnergy1)<epsE)
		{
			if(ComputeDifference(atoms1,tAtoms)<epsT)
			{
				lowerStep = intCurrStep;
				//fprintf(fp, "***Relaxes to minimum 1 at index " BIGINT_FORMAT ", with difference %f\n",
						//intCurrStep, ComputeDifference(atoms1,tAtoms));
			}
		}
		else if((minEnergy3-minEnergy1)<epsE)
		{
			if(ComputeDifference(atoms2,tAtoms)<epsT)
			{
				higherStep = intCurrStep;
				//fprintf(fp, "***Relaxes to minimum 2 at index " BIGINT_FORMAT ", with difference %f\n",
						//intCurrStep, ComputeDifference(atoms2,tAtoms));
			}
		}
		else{
			//fprintf(fp, "***New minimum found at index " BIGINT_FORMAT ", with energy %f\n", intCurrStep, minEnergy3);
			extraList[extraMin] = intCurrStep;
			extraMin++;
			higherStep = intCurrStep;
		}
		fprintf(fp, "UPDATE (" BIGINT_FORMAT ", %f): lower=" BIGINT_FORMAT ", higher=" BIGINT_FORMAT "\n", iSteps,
				log2(nsteps), lowerStep, higherStep);
		iSteps++;
		if(higherStep-lowerStep<=1) break;
	}

	WriteTLS(intCurrStep,atoms1,tAtoms,minEnergy1,minEnergy3);


	//Deletes readInput, to prevent memory leaks.
	delete readInput;

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

void Bisection::InitializeMinimize()
{
	update->etol = 1.0e-4;
	update->ftol = 1.0e-6;
	update->nsteps = 100;
	update->max_eval = 1000;
	update->whichflag = 2;
	update->beginstep = update->firststep = update->ntimestep;
	update->endstep = update->laststep = update->firststep + update->nsteps;
}

double Bisection::CallMinimize()
{
	lmp->init();
	update->minimize->setup();

	timer->init();
	timer->barrier_start();
	update->minimize->run(update->nsteps);
	timer->barrier_stop();

	update->minimize->cleanup();

	Finish finish(lmp);
	finish.end(1);

	//update->whichflag = 0;
	update->firststep = update->laststep = 0;
	update->beginstep = update->endstep = 0;
	return update->minimize->efinal;
}

int Bisection::UpdateDumpArgs(bigint currStep, char *charCurrStep)
{
	std::string strCurrStep = std::to_string((long long)currStep);
	std::strcpy(charCurrStep,strCurrStep.c_str());
	return currStep;
}

//Calculates the difference between two minima.  Now, it finds the rms difference between vectors.
//**Can definitely be optimized by using BLAS libraries to find vector differences
//and then taking the dot product between the vectors.
double Bisection::ComputeDifference(double** x1,double** x2)
{
	double DiffSq=0.0;
	double EnergyWeighting = 100;
	double* m = atom->mass;
	int* type = atom->type;
	for(int i=0;i<atom->natoms;i++)
	{
		for(int j=0;j<domain->dimension;j++)
		{
			DiffSq = DiffSq + m[type[i]]*m[type[i]]*(x1[i][j]-x2[i][j])*(x1[i][j]-x2[i][j]);
		}
	}
	//fprintf(screen, "The difference is %f \n",DiffSq);
	return sqrt(DiffSq);
}

void Bisection::OpenTLS()
{
	std::string strFile = "TLS.dump";
	char *charFile = new char[20];
	std::strcpy(charFile,strFile.c_str());
	int me;
	MPI_Comm_rank(world,&me);
	if(me==0){
		fp = fopen(charFile,"a");
	}
	return;
}

void Bisection::WriteTLS(bigint step, double** x1, double** x2, double E1, double E2)
{
	double dist = ComputeDifference(x1,x2);
	double Ediff = E2 - E1;
	fprintf(fp, BIGINT_FORMAT "\t%f\t%f \n",
			step, Ediff, dist);
	return;
}
