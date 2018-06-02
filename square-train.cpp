
#include <iostream>
#include <vector>
#include <iterator>

#include <caffe/blob.hpp>
#include <caffe/util/upgrade_proto.hpp>
#include <caffe/net.hpp>
#include <caffe/solver.hpp>
#include <caffe/layers/input_layer.hpp>

#include <glog/logging.h>


// following: https://medium.com/@shiyan/caffe-c-helloworld-example-with-memorydata-input-20c692a82a22
// but trying to create a square function, not an xor


int main(int argc, char *argv[])
{
   using namespace std;
   using namespace caffe;
   using namespace boost;
   using boost::shared_ptr;


   cout << "Hello" << endl;
   google::InitGoogleLogging(argv[0]);

   vector<int> dim(2,0); // number of observations

   #define BATCH_SIZE 100

   dim[0] = BATCH_SIZE;  // batch size
   dim[1] =  1;          // elements per batch

   Blob<float>   data(dim), label(dim) ;
   vector<Blob<float> *> blobList;
   blobList.push_back(&data);
   blobList.push_back(&label);


   // Lets create a Solver
   SolverParameter mySolverParams;
   ReadSolverParamsFromTextFileOrDie("./myModel.proto", &mySolverParams);

   Solver<float> *s = SolverRegistry<float>::CreateSolver(mySolverParams);
   shared_ptr<Solver<float>> mySolver(SolverRegistry<float>::CreateSolver(mySolverParams));

   shared_ptr<Net<float>> n = mySolver->net();

   // setup dimensions of input layer
   n->layer_by_name("vishnu-layer-1").get()->LayerSetUp(blobList, blobList);

   const vector<Blob<float> *> & inputBlob = n->input_blobs();

   // train the neural net
   float *d = inputBlob[0]->mutable_cpu_data(), *l = inputBlob[1]->mutable_cpu_data();

   for (int j=0; j< BATCH_SIZE; j++) {
      d[j] = rand() % 100;
      l[j] = d[j]*d[j];
      cout << "(" << d[j] << "," << l[j] << ")" << endl;
   }
   cout << endl;

   mySolver->Solve();

   cout << "Finished training.." << endl;

   // lets get the output
   cout << "predicted-squre-asum-data is = " << n->blob_by_name("predicted-square")->asum_data() << endl;
   cout << "myloss-output-asum-data is = "   << n->blob_by_name("myloss-output")->asum_data() << endl;

   return 0;
}
