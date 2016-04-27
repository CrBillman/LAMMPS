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

using namespace LAMMPS_NS;

#define MAXLINE 2048

/* ---------------------------------------------------------------------- */

Bisection::Bisection(LAMMPS *lmp) : Pointers(lmp)
{
	nAtomArrays = 0;
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
	double** lAtoms = InitAtomArray();
	double** hAtoms = InitAtomArray();
	double** tAtoms = InitAtomArray();
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

	int nInput = ConvertToChar(readInput, strInput);

	//Creates a ReadDump class, and then has it read the appropriate timestep, using the parsed input string.
	
	ReadDump *bisRead = new ReadDump(lmp);
	
//Good to here
	bisRead->command(nInput, readInput);
	//Minimizes the atomic configuration and then stores the energy in lEnergyMin.
	
	lEnergyMin = CallMinimize();

	UpdateAtoms(tAtoms);
	CopyAtoms(lAtoms,tAtoms);
	readInput[1] = &charCurrStep[0];
	intCurrStep = UpdateDumpArgs(nsteps, charCurrStep);
	bisRead->command(nInput, readInput);

	hEnergyMin = CallMinimize();
	UpdateAtoms(tAtoms);
	CopyAtoms(hAtoms,tAtoms);

	//Test Functions
	//TestComputeDifference();
	//TestMinimize(nsteps, bisRead, nInput, readInput);
	if(me==0)
	{
		if(ComputeDifference(lAtoms,hAtoms)<epsT)
		{
			fprintf(fp,"End-points for bisection have same minimum.  Bisection may fail.\n");
		}
	}

	bigint iSteps = 0;

	while(iSteps<maxSteps)
	{
		intCurrStep = UpdateDumpArgs((higherStep-lowerStep)/2+lowerStep,charCurrStep);
		tEnergyMin = CallMinimize();
		UpdateAtoms(tAtoms);
		eDiff = fabs(tEnergyMin-lEnergyMin);
		lDistDiff = ComputeDifference(lAtoms,tAtoms); 
		hDistDiff = ComputeDifference(hAtoms,tAtoms);
		if((lDistDiff<hDistDiff)&&(lDistDiff<epsT))
		{
			if(me==0)  fprintf(fp, "UPDATE-Match L (%f, %f, %f, %f): lower=" BIGINT_FORMAT ", higher=" BIGINT_FORMAT "\n", lEnergyMin,
                                        hEnergyMin, eDiff, lDistDiff, intCurrStep, higherStep);
			lowerStep = intCurrStep;
			CopyAtoms(lAtoms, tAtoms);
			lEnergyMin = tEnergyMin;
		}
		else{
			if(hDistDiff<epsT)
			{
				if(me==0)  fprintf(fp, "UPDATE-Match U (%f, %f, %f, %f): lower=" BIGINT_FORMAT ", higher=" BIGINT_FORMAT "\n", lEnergyMin,
                                        hEnergyMin, eDiff, hDistDiff, lowerStep, intCurrStep);
				higherStep = intCurrStep;
				CopyAtoms(hAtoms, tAtoms);
				hEnergyMin = tEnergyMin;
			}
			else{
				if(me==0)  fprintf(fp, "UPDATE-Match U (%f, %f, %f, %f): lower=" BIGINT_FORMAT ", higher=" BIGINT_FORMAT "\n", lEnergyMin,
					hEnergyMin, eDiff, hDistDiff, lowerStep, intCurrStep);
				higherStep = intCurrStep;
				CopyAtoms(hAtoms, tAtoms);
				hEnergyMin = tEnergyMin;
			}
		}
		eDiff = fabs(hEnergyMin-lEnergyMin);
		iSteps++;
		if(higherStep-lowerStep<=1) break;
	}

	if(me==0) WriteTLS(intCurrStep,lAtoms,hAtoms,lEnergyMin,hEnergyMin);
	StoreAtoms(lAtoms, hAtoms);
	//WriteAtoms(lAtoms, hAtoms);
	if(me==0) fprintf(screen, "The first atom position is %f\n",lAtoms[0][0]);

	//Deletes readInput and atom arrays, to prevent memory leaks.
	
	delete bisRead;
	memory->sfree(readInput);
	DeleteAtomArray(lAtoms);
	DeleteAtomArray(hAtoms);
	DeleteAtomArray(tAtoms);	
	if(me==0) fclose(fp);

    	if (atom->map_style!=0) {
      		atom->map_delete();
	      	atom->map_style = 0;
    	}

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
	char **newarg = new char*[4];
	newarg[0] = (char *) "1.0e-4";
	newarg[1] = (char *) "1.0e-6";
	newarg[2] = (char *) "100";
	newarg[3] = (char *) "1000";
	Minimize* rMin = new Minimize(lmp);
	rMin->command(4, newarg);
	delete rMin;
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
	//std::string strCurrStep = std::to_string((long long)currStep);
	std::ostringstream oss;
	oss << (long long)currStep;
	std::string strCurrStep = oss.str();
	std::strcpy(charCurrStep,strCurrStep.c_str());
	return currStep;
}

//Calculates the difference between two minima.  Now, it finds the rms difference between vectors.
//**Can definitely be optimized by using BLAS libraries to find vector differences
//and then taking the dot product between the vectors.
double Bisection::ComputeDifference(double** x1,double** x2)
{
	double DiffSq;
	double TotDiff = 0.0;
	double EnergyWeighting = 100;
	double* m = atom->mass;
	int* type = atom->type;
	double totMass = 0.0;
	double* halfDim = domain->prd_half;
	double Diff;

	for(int i=0;i<atom->natoms;i++)
	{
		DiffSq = 0.0;
		totMass = totMass + m[type[i]];
		for(int j=0;j<domain->dimension;j++)
		{
			Diff = (x1[i][j]-x2[i][j]);
			if(Diff<(-halfDim[j]))
			{
				Diff = Diff + 2*halfDim[j];
			}
			else if(Diff>halfDim[j])
			{
				Diff = Diff - 2*halfDim[j];
			}
			DiffSq = DiffSq + Diff*Diff;
		}
		TotDiff = TotDiff + m[type[i]]*DiffSq;
	}
	TotDiff = TotDiff/totMass;
	return sqrt(TotDiff);
}

void Bisection::TestComputeDifference()
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

	Diff = ComputeDifference(Atoms1, Atoms2);
	if(fabs(sqrt(3)-Diff)<1E-3)
	{
		if(me==0) fprintf(screen, "ComputeDifference passes test 1.\n");
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

	Atoms2[0][0] = 1;
        Diff = ComputeDifference(Atoms1, Atoms2);
        if(fabs(sqrt(m[type[0]]/MassTot)-Diff)<1E-3)
        {
                if(me==0) fprintf(screen, "ComputeDifference passes test 2.\n");
        }
        else
        {
                if(me==0) fprintf(screen, "Computedifference fails test 2.  Expected Diff==%f, but got %f.\n", sqrt(m[type[0]]/MassTot), Diff);
        }

	//Third test sets the positions to be at the edges of the unit cell.  This should give 0 displacement if the edges are treated correctly.
	for(int j=0;j<domain->dimension;j++)
	{
		Atoms1[0][j] = -(domain->prd_half[j]);
		Atoms2[0][j] = domain->prd_half[j];
	}

        Diff = ComputeDifference(Atoms1, Atoms2);
        if(fabs(Diff)<1E-3)
        {
                if(me==0) fprintf(screen, "ComputeDifference passes test 3.\n");
        }
        else
        {
                if(me==0) fprintf(screen, "Computedifference fails test 3.  Expected Diff==%f, but got %f.\n", 0.0, Diff);
        }

	DeleteAtomArray(Atoms1);
	DeleteAtomArray(Atoms2);
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
	double dist = ComputeDifference(x1,x2);
	double Ediff = E2 - E1;
	fprintf(fp, BIGINT_FORMAT "\t%f\t%f \n",
			step, Ediff, dist);
	return;
}

double** Bisection::InitAtomArray()
{
	double *data = (double *)malloc(3*atom->natoms*sizeof(double));
	double** atomArray = (double **)malloc(atom->natoms*sizeof(double *));
	for(int i=0; i<atom->natoms; i++)
	{
		atomArray[i] = &(data[3*i]);

	}
	return atomArray;
}

void Bisection::DeleteAtomArray(double** atomArray)
{
	memory->sfree(atomArray[0]);
	memory->sfree(atomArray);
	return;
}

void Bisection::CopyAtoms(double** copyArray, double** templateArray)
{
	for(int i=0;i<atom->natoms;i++)
	{
		for(int j=0;j<domain->dimension;j++)
		{
			copyArray[i][j] = templateArray[i][j];
		}
	}
}

void Bisection::MappedCopyAtoms(double** copyArray, double** templateArray)
{
	int m;
        for(int i=1;i<=atom->natoms;i++)
        {
		m = atom->map(i);
		if(m>=0 && m<atom->nlocal)
		{
			for(int j=0;j<domain->dimension;j++)
			{
				copyArray[m][j] = templateArray[i-1][j];
			}
		}
        }
}

void Bisection::StoreAtoms(double** pos1, double** pos2)
{
	char **newarg = new char*[5];
        int me;
        MPI_Comm_rank(world,&me);

	newarg[0] = (char *) "TLS1";
	newarg[1] = (char *) "all";
	newarg[2] = (char *) "STORE";
	newarg[3] = (char *) "0";
	newarg[4] = (char *) "3";
	modify->add_fix(5,newarg);
	FixStore *TLS1 = (FixStore *) modify->fix[modify->nfix-1];
	MappedCopyAtoms(TLS1->astore,pos1);
	
	newarg[0] = (char *) "TLS2";
	modify->add_fix(5,newarg);
        FixStore *TLS2 = (FixStore *) modify->fix[modify->nfix-1];
        MappedCopyAtoms(TLS2->astore,pos2);

	delete [] newarg;
}

/*void Bisection::WriteAtoms(double** pos1, double** pos2)
{
        char **newarg = new char*[5];
        int me;
        MPI_Comm_rank(world,&me);

        newarg[0] = (char *) "TLS1";
        newarg[1] = (char *) "all";
        newarg[2] = (char *) "atom";
        newarg[3] = (char *) "0";
        newarg[4] = (char *) "TLS1.dump";
        CopyAtoms(atom->x,pos1);
	Dump *TLS1 = new Dump_Atom(lmp);
	TLS1->command(5,newarg);
	delete TLS1;

        newarg[0] = (char *) "TLS2";
	newarg[4] = (char *) "TLS2.dump";
        CopyAtoms(atom->x,pos2);
	Dump *TLS2 = new Dump_Atom(lmp);
	TLS2->command(5,newarg);
	delete TLS2;

        delete [] newarg;
	return;
}*/

void Bisection::UpdateAtoms(double** currAtoms)
{
	double** atomPtr = atom->x;
	double* tArr1 = new double[domain->dimension*atom->natoms];
	double* tArr2 = new double[domain->dimension*atom->natoms];
	int *mask = atom->mask;
        int me;
	int m;
        MPI_Comm_rank(world,&me);
	//int groupbit = group->bitmask[igroup];
	if (atom->map_style == 0) {
		atom->nghost = 0;
		atom->map_init();
		atom->map_set();
	}
	//std::cout << "Number of local atoms: " << me << "\t" << atom->nlocal << std::endl;
	for(int i = 1; i <= atom->natoms; i++)
	{
		m = atom->map(i);
		//if (mask[m]) {
		if(m>=0 && m<atom->nlocal)
		{
			//std::cout << me << "\t" << i << "\t" << m << "\t" << atomPtr[m][0] << std::endl;
			//std::cout << i << "\t" << atomPtr[m][0] << "\t" << atomPtr[m][1] << "\t" << atomPtr[m][2] << std::endl;
			for(int j = 0; j < domain->dimension; j++)
			{
				tArr1[3*(i-1)+j] = atomPtr[m][j];
			}
		}
		else
		{
			for(int j = 0; j < domain->dimension; j++)
                        {
                                tArr1[3*(i-1)+j] = 0.0;
			}
		}
	}

	//if(me==0) fprintf(screen, "Finshed the loop!\n");
	
	MPI_Allreduce(tArr1,tArr2,(3*atom->natoms),MPI_DOUBLE,MPI_SUM,world);
	
	for(int i=0; i < atom->natoms; i++)
	{
		for(int j=0; j < domain->dimension; j++)
		{
			currAtoms[i][j] = tArr2[3*i+j];
		}
	}
	
	/*if(me==0)
	{
		for(int i=0; i< atom->natoms; i++)
		{
			fprintf(screen,"Atom %d, %f\t%f\t%f\n",i,currAtoms[i][0],currAtoms[i][1],currAtoms[i][2]);
		}
	}*/
	

	return;
}
