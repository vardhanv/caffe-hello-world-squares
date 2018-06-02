
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

   vector<int> batchSz(2,0); // number of observations

   cout << "Hello" << endl;
   google::InitGoogleLogging(argv[0]);

   #define BATCH_SIZE 1
   batchSz[0] = BATCH_SIZE;  // batch size
   batchSz[1] =  1;          // elements per batch
   
   Blob<float>   data(batchSz), label(batchSz) ;
   vector<Blob<float> *> blobList;
   blobList.push_back(&data);
   blobList.push_back(&label);


   // ---- The test phase --- 
   // lets see if it works

   // create the inference net
   shared_ptr<Net<float>> testNet (new Net<float>("./myNetParam.proto", TEST));
   testNet->set_debug_info(true);

   testNet->CopyTrainedLayersFrom("./myModel_iter_350000.caffemodel");


   float test_data[] =  { 10, 20, 30 , 40 } ;
   float fake_label[] = { 10, 10, 10 , 40 } ;

   testNet->layer_by_name("test-layer-1").get()->LayerSetUp(blobList, blobList);

   vector<Blob<float> *> ib = testNet->input_blobs();

   float * d = ib[0]->mutable_cpu_data(), *l = ib[1]->mutable_cpu_data();

   for(int i=0; i < sizeof(test_data)/sizeof(*test_data); i++) {
      d[0] = test_data[i] ;
      l[0] = fake_label[i] ;

      testNet->Forward();

      // lets get the output
      cout << "Square of asum_data " << test_data[i] << " is = " << testNet->blob_by_name("predicted-square")->asum_data()  << endl;
   }

   return 0;
}

