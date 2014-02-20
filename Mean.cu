
/*
 * Description:
 *    this function computes the mean along the innermost dimension
 *    Nd input, (N-1)d output
 */
__global__ void mean_output(float *input, float *output,
                           long nrows, long ncols)
{
  // output offset:
  long o = threadIdx.x + blockDim.x * blockIdx.x;
  if (o >= nrows) return;

  // input offset:
  long i = o * ncols;

  // move pointers
  input = input + i;

  // compute mean:
  float mean = input[0];
  long ii;
  for (ii=1; ii<ncols; ii++) {
      float val = input[ii];
      mean = mean + (val - mean)/ncols;
  }

  // store
  output[o] = mean;
}

static int cunn_Mean_updateOutput(lua_State *L)
{
  printf("calling our own mean.updateOutput\n");
  THCudaTensor *input = (THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");
  int dimension = luaT_getfieldcheckint(L, 1, "dimension")-1;
  THCudaTensor *output = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "output", "torch.CudaTensor");

  luaL_argcheck(L, dimension >= 0 && dimension < input->nDimension, 2, "dimension out of range");
  luaL_argcheck(L, dimension == input->nDimension-1, 2, "only supported dimension is innermost (CUDA kernel only)");

  input = THCudaTensor_newContiguous(input);

  THLongStorage *dim = THLongStorage_newWithSize(input->nDimension);
  long i;
  for(i = 0; i < input->nDimension; i++)
    dim->data[i] = input->size[i];
  dim->data[dimension] = 1;
  THCudaTensor_resize(output, dim, NULL);
  THLongStorage_free(dim);

  float *input_data = THCudaTensor_data(input);
  float *output_data = THCudaTensor_data(output);

  long nrows = THCudaTensor_nElement(output);
  long ncols = input->size[dimension];

  // cuda blocks & threads:
  long nthreads = 256;
  long nblocks = ceil((float)nrows / nthreads);
  dim3 blocks(nblocks);
  dim3 threads(nthreads);

  // kernel:
  mean_output <<<blocks, threads>>> (input_data, output_data, nrows, ncols);

  // check for errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in Mean.updateOutput: %s\n", cudaGetErrorString(err));
    THError("aborting");
  }

  // final cut:
  THCudaTensor_free(input); 
  THCudaTensor_select(output, NULL, dimension, 0);

  return 1;
}

static int cunn_Mean_updateGradInput(lua_State *L)
{
  THCudaTensor *input = (THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *gradOutput = (THCudaTensor *)luaT_checkudata(L, 3, "torch.CudaTensor");
  int dimension  = luaT_getfieldcheckint(L, 1, "dimension")-1;
  THCudaTensor *gradInput  = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "gradInput", "torch.CudaTensor");

  long nrows = THCudaTensor_nElement(gradOutput);
  long ncols = gradInput->size[dimension];

  THCudaTensor_resizeAs(gradInput, input);
  THCudaTensor_copy(gradInput, gradOutput);
  THCudaTensor_mul(gradInput, 1./ncols);

  return 1;
}

static const struct luaL_Reg cunn_Mean__ [] = {
  {"Mean_updateOutput", cunn_Mean_updateOutput},
  {"Mean_updateGradInput", cunn_Mean_updateGradInput},
  {NULL, NULL}
};

static void cunn_Mean_init(lua_State *L)
{
  printf("initing our own mean version\n");
  luaT_pushmetatable(L, "torch.CudaTensor");
  luaT_registeratname(L, cunn_Mean__, "nn");
  lua_pop(L,1);
}
