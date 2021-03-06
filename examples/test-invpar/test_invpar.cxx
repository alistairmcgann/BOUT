/*
 * Test parallel inversion routines
 *
 */

#include <bout.hxx>
#include <boutmain.hxx>

#include <invert_parderiv.hxx>
#include <field_factory.hxx>
#include <utils.hxx>

int physics_init(bool restarting) {
  InvertPar *inv = InvertPar::Create();
  FieldFactory f;

  // Get options
  Options *options = Options::getRoot();
  string acoef, bcoef, func;
  options->get("acoef", acoef, "1.0");
  options->get("bcoef", bcoef, "-1.0");
  options->get("input", func, "sin(2*y)");
  BoutReal tol;
  OPTION(options, tol, 1e-5);
  
  Field2D A = f.create2D(acoef);
  Field2D B = f.create2D(bcoef);

  inv->setCoefA(A);
  inv->setCoefB(B);
  
  Field3D input = f.create3D(func);
  Field3D result = inv->solve(input);
  mesh->communicate(result);
  
  Field3D deriv = A*result + B * Grad2_par2(result);
  
  // Check the result
  int passed = 1;
  for(int y=2;y<mesh->ngy-2;y++) {
    for(int z=0;z<mesh->ngz-1;z++)  {
      output.write("result: [%d,%d] : %e, %e, %e\n", y,z, input[2][y][z], result[2][y][z], deriv[2][y][z]);
      if(abs(input[2][y][z] - deriv[2][y][z]) > tol)
	passed = 0;
    }
  }
  
  int allpassed;
  MPI_Allreduce(&passed, &allpassed, 1, MPI_INT, MPI_MIN, BoutComm::get());
  
  output << "******* Parallel inversion test case: ";
  if(allpassed) {
    output << "PASSED" << endl;
  }else 
    output << "FAILED" << endl;

  // Saving state to file
  SAVE_ONCE(allpassed);
  
  // Write data to file
  dump.write();
  dump.close();
  
  MPI_Barrier(BoutComm::get());
  
  // Send an error code so quits
  return 1;
}

int physics_run(BoutReal t) {
  // Doesn't do anything
  return 1;
}
