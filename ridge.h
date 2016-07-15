/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2026) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef COMMAND_CLASS

CommandStyle(ridge,Ridge)

#else

#ifndef LMP_RIDGE_H
#define LMP_RIDGE_H

#include "pointers.h"
#include <stdlib.h>
#include <string>

namespace LAMMPS_NS {

class Ridge : protected Pointers {
private:
	FILE *fp;
        int nRSteps;
        int nBSteps;
	int nPRelSteps;
	int nMRelSteps;
        double epsT, epsF;
	double eTLS1, eTLS2;
	double **pTLS1, **pTLS2, **pTLSs;
	double *lat1, *lat2, *latS;
	double **hAtoms, **lAtoms, **tAtoms;
	double *hLat, *lLat, *tLat;
	void PerformRidge();
	int LoadPositions();
	void ReadPositions();
        void CopyAtoms(double **, double **);
	void ComparePositions(double **, double **, double **);
        double** InitAtomArray();
        void DeleteAtomArray(double **);
        double CallMinimize();
        double ComputeDistance(double **,double **);
	int InitHessianCompute();
	void CopyLatToBox(double *);
	void CopyBoxToLat(double *);
	void CopyLatToLat(double *, double *);
	void TestComputeDistance();
        void WriteTLS(double, double, double);
        void OpenTLS();
	void TestBisect();
	void BisectPositions(double **, double **, double **);
	void ToAtomMapping(double **);
	void PartialRelax(double **, double **);
	void ConvertIntToChar(char *, int);
	void UpdateMapping();
	void InitAtomArrays();
	void ResetBox();
	bool CheckSaddle(double **);
public:
	Ridge(class LAMMPS *);
	void command(int, char **);
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

*/
