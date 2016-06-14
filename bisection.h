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

CommandStyle(bisection,Bisection)

#else

#ifndef LMP_BISECTION_H
#define LMP_BISECTION_H

#include "pointers.h"
#include "read_dump.h"
#include <stdlib.h>
#include <string>

namespace LAMMPS_NS {

class Bisection : protected Pointers {
private:
	int nAtomArrays;
	FILE *fp;
	int inputSetFlag;
	double epsT;
        double** lAtoms;
        double** hAtoms;
        double** tAtoms;
	void BisectionFromMD(bigint, char*);
	int ConvertToChar(char **, std::string);
	double CallMinimize();
	int UpdateDumpArgs(bigint, char*);
	double ComputeDistance(double**,double**);
	void WriteTLS(bigint, double**, double**, double, double);
	void OpenTLS();
	void TestMinimize(bigint, ReadDump*, int, char**);
	void TestComputeDistance();
	void StoreAtoms(double**, double**);
	void WriteAtoms(double**, double**);
        void CopyAtoms(double**, double**);
        void MappedCopyAtoms(double**, double**);
        void InitAtomArrays();
        void DeleteAtomArray(double**);
public:
	Bisection(class LAMMPS *);
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
