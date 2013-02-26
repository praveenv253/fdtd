/* Program to simulate propagation of a gaussian wave in dipole coordinates */

#include <cstdio>
#include <iostream>
#include <fstream>

#define Sc_nu 0.2f
#define Sc_mu 0.2f
#define imp0 377.0f
#define EPSILON_R 1
#define MU_R 1

static void
check_error(cudaError_t err, const char *file, int line) {
    if(err != cudaSuccess) {
        std::cout<<cudaGetErrorString(err)<<" in "<<file;
        std::cout<<" at line "<<line<<"\n";
        exit(EXIT_FAILURE);
    }
}
#define _(err) (check_error(err, __FILE__, __LINE__))

// These are currently fixed, but they should be user-modifiable
const int SIZE_NU = 128;
const int SIZE_MU = 128;
const float NU_MAX = 1.0;
const float NU_MIN = NU_MAX / SIZE_NU;
const float MU_MAX = 1.0;
const float MU_MIN = MU_MAX / SIZE_MU;
const int MAXTIME = 100;
const int SOURCE_LOCATION = 55 * SIZE_MU + 55;
const int RI = 1;

// Define structure to be passed to kernel
struct Data {
    float *r;
    float *sin_theta;
    float *h_nu;
    float *h_phi;
    float *h_mu;
    float *E_nu;
    float *E_phi;
    float *H_nu;
    float *H_phi;
    float *H_mu;
    struct Data *d;
};

// Compute r, theta from nu, mu
__global__ void
compute_r_theta(float *r_matrix, float *sin_theta_matrix)
{
    // Get thread position
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    
    if(x < SIZE_NU && y < SIZE_MU) {
        float nu = NU_MIN + x * (NU_MAX - NU_MIN) / SIZE_NU;
        float mu = MU_MIN + y * (MU_MAX - MU_MIN) / SIZE_MU;
        
        // Note: nu and mu are just x and y. This obviates the need for the
        // nu_matrix and mu_matrix
        
        // Find position to store r and theta
        float *r = r_matrix + x*SIZE_MU + y;
        float *sin_theta = sin_theta_matrix + x*SIZE_MU + y;
        
        // Invert nu-mu to r-theta
        float alpha = (256.0f * mu * mu) / (27 * (nu * nu * nu * nu));
        float beta = __powf((1 + __powf(1 + alpha, 0.5f)), (2.0f/3));
        float gamma = __powf(alpha, (1.0f/3));
        float zeta = __powf( ((beta*beta + beta*gamma + gamma*gamma) / beta), 
                             1.5f
                           ) / 2;
        (*r) = 4 * zeta / (nu * (1 + zeta) * (1 + __powf(2*zeta - 1, 0.5f)));
        (*sin_theta) = __powf((*r) * nu, 0.5f);
        
        // These may be needed for plotting. Decide what to do with them later.
        // float delta = __powf( (4 - 3*(*sin_theta)*(*sin_theta)), 0.5 );
        // float gridx = (*r) * __powf(1 - (*sin_theta)*(*sin_theta), 0.5);
        // float gridy = (*r) * sin_theta;
    }
}

// In the computation of the h-matrices, the existing matrices can be 
// overwritten. Beyond this point, only the h-matrices are actually used.
__global__ void
compute_h(float *h_nu_matrix, float *h_phi_matrix, float *h_mu_matrix, 
          float *r_matrix, float *sin_theta_matrix)
{
    // Get thread position
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    // Check limits on x and y
    if(x >= SIZE_NU || y >= SIZE_MU) return;
    
    float *h_nu = h_nu_matrix + x*SIZE_MU + y;
    float *h_phi = h_phi_matrix + x*SIZE_MU + y;
    float *h_mu = h_mu_matrix + x*SIZE_MU + y;
    float r = r_matrix[x*SIZE_MU + y];
    float sin_theta = sin_theta_matrix[x*SIZE_MU + y];
    
    float delta = __powf( (4 - 3*sin_theta*sin_theta), 0.5 );
    
    // Assume RI is #defined somewhere.
    (*h_nu) = (r * r) / (RI * sin_theta * delta);
    (*h_phi) = r * sin_theta;
    (*h_mu) = (r * r * r) / (RI * RI * delta);
}
// Note: 64KB of const memory => 64K/3 ~ 21K => 5461 elements per h-matrix
//                            => matrix side length = 73 => abysmal
// That is, 64KB allows only a maximum simulation size of 73x73.
// And this is excluding epsilon and mu matrices which will also come into the 
// mix later on.

// Update for H_nu
__global__ void
update_H_nu(struct Data *d) {
    // Get thread position
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    // Check limits on x and y
    if(x >= SIZE_NU || y >= SIZE_MU-1) return;
    
    // Compute coefficients
    int index = x*(SIZE_MU-1) + y;
    float cH = 1;
    float h_phi_prev = d->h_phi[index];
    float h_phi_next = d->h_phi[index + 1];
    float h_phi_avg = (h_phi_prev + h_phi_next) / 2;
    float h_mu_avg = (d->h_mu[index] + d->h_mu[index + 1]) / 2;
    // Sc_mu and imp0 should be __device__ const float somewhere
    // MU_R is #defined somewhere
    float cE = Sc_mu / (imp0 * MU_R * h_phi_avg * h_mu_avg);
    
    // Now for the actual update
    float *H_nu = d->H_nu + index;
    float E_phi_prev = d->E_phi[index];
    float E_phi_next = d->E_phi[index + 1];
    (*H_nu) = cH * (*H_nu) + cE * (  h_phi_next * E_phi_next
                                   - h_phi_prev * E_phi_prev );
}

// Update for H_phi
__global__ void
update_H_phi(struct Data *d) {
    // Get thread position
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    // Check limits on x and y
    if(x >= SIZE_NU-1 || y >= SIZE_MU-1) return;
    
    // Compute coefficients
    // Note: column width of H_phi is SIZE_MU-1, but column width of h-matrices
    // are all SIZE_MU.
    int index = x*(SIZE_MU-1) + y;
    float cH = 1;
    float h_nu = (  d->h_nu[index] + d->h_nu[index + SIZE_MU + 1]
                  + d->h_nu[index + 1] + d->h_nu[index + SIZE_MU] ) / 4;
    float h_mu = (  d->h_mu[index] + d->h_mu[index + SIZE_MU + 1]
                  + d->h_mu[index + 1] + d->h_mu[index + SIZE_MU] ) / 4;
    float h_nu_fwd_avg = (  d->h_nu[index + SIZE_MU + 1]
                          + d->h_nu[index + 1]           ) / 2;
    float h_nu_bwd_avg = (d->h_nu[index] + d->h_nu[index + SIZE_MU]) / 2;
    float cE = - Sc_mu / (imp0 * MU_R * h_nu * h_mu);

    // Final update equation
    float *H_phi = d->H_phi + index;
    float E_nu_fwd = d->E_nu[index + 1];
    float E_nu_bwd = d->E_nu[index];
    (*H_phi) = cH * (*H_phi) + cE * (  h_nu_fwd_avg * E_nu_fwd
                                     - h_nu_bwd_avg * E_nu_bwd );
}

// Update for H_mu
__global__ void
update_H_mu(struct Data *d) {
    // Get thread position
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    // Check limits on x and y
    if(x >= SIZE_NU-1 || y >= SIZE_MU) return;
    
    // Compute coefficients
    int index = x*SIZE_MU + y;
    float cH = 1;
    float h_nu_avg = (d->h_nu[index] + d->h_nu[index + SIZE_MU]) / 2;
    float h_phi_prev = d->h_phi[index];
    float h_phi_next = d->h_phi[index + SIZE_MU];
    float h_phi_avg = (h_phi_prev + h_phi_next) / 2;
    // Sc_nu also has to be __device__ const float somewhere.
    float cE = - Sc_nu / (imp0 * MU_R * h_nu_avg * h_phi_avg);
    
    // Final update equation
    float *H_mu = d->H_mu + index;
    float E_phi_prev = d->E_phi[index];
    float E_phi_next = d->E_phi[index + SIZE_MU];
    (*H_mu) = cH * (*H_mu) + cE * (  h_phi_next * E_phi_next
                                   - h_phi_prev * E_phi_prev );
}

// Update for E_nu
__global__ void
update_E_nu(struct Data *d) {
    // Get thread position
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    // Check limits on x and y
    if(x >= SIZE_NU-1 || y == 0 || y >= SIZE_MU-1) return;
    
    // Compute coefficients
    int index = x*SIZE_MU + y;
    float cE = 1;
    float h_phi_avg = (d->h_phi[index] + d->h_phi[index + SIZE_MU]) / 2;
    float h_mu_avg = (d->h_mu[index] + d->h_mu[index + SIZE_MU]) / 2;
    float h_phi_fwd_avg = (  d->h_phi[index]
                           + d->h_phi[index + SIZE_MU + 1]
                           + d->h_phi[index + SIZE_MU]
                           + d->h_phi[index + 1]           ) / 4;
    float h_phi_bwd_avg = (  d->h_phi[index]
                           + d->h_phi[index + SIZE_MU - 1]
                           + d->h_phi[index + SIZE_MU]
                           + d->h_phi[index - 1]           ) / 4;
    // EPSILON_R to be #defined somewhere
    float cH = - Sc_mu * imp0 / (EPSILON_R * h_phi_avg * h_mu_avg);
    
    // Final update equation
    float *E_nu = d->E_nu + index;
    // Note: Column width of H_phi is only SIZE_MU-1. Need to use appropriate
    // column sizes for each matrix
    float H_phi_next = d->H_phi[x*(SIZE_MU-1) + y];
    float H_phi_prev = d->H_phi[x*(SIZE_MU-1) + y - 1];
    (*E_nu) = cE * (*E_nu) + cH * (  h_phi_fwd_avg * H_phi_next
                                   - h_phi_bwd_avg * H_phi_prev );
}

// Update for E_phi
__global__ void
update_E_phi(struct Data *d, int t) {
    // Get thread position
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    // Check limits on x and y
    if(x == 0 || x >= SIZE_NU-1 || y == 0 || y >= SIZE_MU-1) return;
    
    // Compute coefficients
    int index = x*SIZE_MU + y;
    float cE = 1;
    float h_nu = d->h_nu[index];
    float h_mu = d->h_mu[index];
    float h_nu_fwd_avg = (d->h_nu[index + 1] + h_nu) / 2;
    float h_nu_bwd_avg = (h_nu + d->h_nu[index - 1]) / 2;
    float h_mu_fwd_avg = (d->h_mu[index + SIZE_MU] + h_mu) / 2;
    float h_mu_bwd_avg = (h_mu + d->h_mu[index - SIZE_MU]) / 2;
    float cH_nu = Sc_mu * imp0 / (EPSILON_R * h_nu * h_mu);
    float cH_mu = - Sc_nu * imp0 / (EPSILON_R * h_nu * h_mu);
    
    // Final update equation
    float *E_phi = d->E_phi + index;
    // Need to use appropriate column sizes for each matrix
    float H_nu_next = d->H_nu[x*(SIZE_MU-1) + y];
    float H_nu_prev = d->H_nu[x*(SIZE_MU-1) + y - 1];
    float H_mu_next = d->H_mu[index];
    float H_mu_prev = d->H_mu[index - SIZE_MU];
    (*E_phi) =   cE * (*E_phi)
               + cH_nu * (h_nu_fwd_avg * H_nu_next - h_nu_bwd_avg * H_nu_prev)
               + cH_mu * (h_mu_fwd_avg * H_mu_next - h_mu_bwd_avg * H_mu_prev);

    // Needs definition of SOURCE_LOCATION
    if(index == SOURCE_LOCATION)
        E_phi[index] = __expf(-(t-30) * (t-30) / 100.0);
}

int main(int argc, char **argv)
{
    struct Data data;
    
    std::cout<<"Program started\n";
    
    // Declare r, theta matrices
    _(cudaMalloc((void **)&data.r, SIZE_NU * SIZE_MU * sizeof(float)));
    _(cudaMalloc((void **)&data.sin_theta, SIZE_NU * SIZE_MU * sizeof(float)));
    
    std::cout<<"Allocated memory for r and theta matrices\n";
    
    cudaStream_t stream1;
    _(cudaStreamCreate(&stream1));
    
    // Launch kernel to compute r and theta
    dim3 threads(16, 16);
    dim3 blocks(SIZE_NU / 16, SIZE_MU / 16);
    compute_r_theta<<<blocks, threads, 0, stream1>>>(data.r, data.sin_theta);
    
    std::cout<<"Done computing r and theta\n";
    
    // Allocate memory for h-matrices
    _(cudaMalloc((void **)&data.h_nu, SIZE_NU * SIZE_MU * sizeof(float)));
    _(cudaMalloc((void **)&data.h_phi, SIZE_NU * SIZE_MU * sizeof(float)));
    _(cudaMalloc((void **)&data.h_mu, SIZE_NU * SIZE_MU * sizeof(float)));
    
    std::cout<<"Allocated memory for h-matrices\n";
    
    // Launch kernel to compute h-matrices
    compute_h<<<blocks, threads, 0, stream1>>>(data.h_nu, data.h_phi, data.h_mu,
                                               data.r, data.sin_theta);
    
    std::cout<<"Done computing h matrices\n";

    // Declare host versions of E and H matrices
    // Might want to use HostAlloc later on so that copying to this can be made
    // asynchronous
    float E_nu[SIZE_NU-1][SIZE_MU] = { 0 };
    float E_phi[SIZE_NU][SIZE_MU] = { 0 };
    // E_mu is zero
    float H_nu[SIZE_NU][SIZE_MU-1] = { 0 };
    float H_phi[SIZE_NU-1][SIZE_MU-1] = { 0 };
    float H_mu[SIZE_NU-1][SIZE_MU] = { 0 };
    
    std::cout<<"Created host field matrices\n";
     
    // Allocate memory on the device for field matrices
    _(cudaMalloc((void **)&data.E_nu, (SIZE_NU-1) * SIZE_MU * sizeof(float)));
    _(cudaMalloc((void **)&data.E_phi, SIZE_NU * SIZE_MU * sizeof(float)));
    _(cudaMalloc((void **)&data.H_nu, SIZE_NU * (SIZE_MU-1) * sizeof(float)));
    _(cudaMalloc((void **)&data.H_phi,
                 (SIZE_NU-1) * (SIZE_MU-1) * sizeof(float)));
    _(cudaMalloc((void **)&data.H_mu, (SIZE_NU-1) * SIZE_MU * sizeof(float)));
    
    std::cout<<"Allocated memory for device field matrices\n";
    
    // Copy host matrices to device in order to set zero
    _(cudaMemcpy(data.E_nu, E_nu, (SIZE_NU-1) * SIZE_MU * sizeof(float),
                 cudaMemcpyHostToDevice));
    _(cudaMemcpy(data.E_phi, E_phi, SIZE_NU * SIZE_MU * sizeof(float),
                 cudaMemcpyHostToDevice));
    _(cudaMemcpy(data.H_nu, H_nu, SIZE_NU * (SIZE_MU-1) * sizeof(float),
                 cudaMemcpyHostToDevice));
    _(cudaMemcpy(data.H_phi, H_phi, (SIZE_NU-1) * (SIZE_MU-1) * sizeof(float),
                 cudaMemcpyHostToDevice));
    _(cudaMemcpy(data.H_mu, H_mu, SIZE_NU * (SIZE_MU-1) * sizeof(float),
                 cudaMemcpyHostToDevice));
    
    std::cout<<"Done setting zero to device field matrices\n";
    
    // Copy pointers to device
    // One-time minor expense for increased ease of access...
    _(cudaMalloc((void **)&data.d, sizeof(struct Data)));
    _(cudaMemcpy(data.d, &data, sizeof(data), cudaMemcpyHostToDevice));
    
    // Start stepping to update E and H
    /* Note: Maybe we should compute the h-matrices also every single time. In
       all probability, considering that memory access can take upto 500 clock
       cycles, computing h-values each time might be faster - only benchmarking
       will tell */
    // Each update equation occurs as a separate kernel
    int t = 0;
    cudaStream_t stream2, stream3;
    _(cudaStreamCreate(&stream2));
    _(cudaStreamCreate(&stream3));
    
    std::cout<<"Starting time stepping...\n";
    
    while(t < MAXTIME) {
        // Launch update kernels
        // Note that H_nu, H_mu and E_phi form one set of independent equations
        // while H_phi and E_nu form another set. Therefore, they can be run in
        // parallel. Moreover, the update of H_nu and H_mu can also be run in
        // parallel.
        update_H_nu<<<blocks, threads, 0, stream2>>>(data.d);
        update_H_phi<<<blocks, threads, 0, stream1>>>(data.d);
        update_H_mu<<<blocks, threads, 0, stream3>>>(data.d);
        update_E_nu<<<blocks, threads, 0, stream1>>>(data.d);
        // E_phi should wait for both streams 2 and 3. We wait only for 3 since
        // it will be forced to wait for 2 if it is to run on 2.
        _(cudaStreamSynchronize(stream3));
        update_E_phi<<<blocks, threads, 0, stream2>>>(data.d, t);
        
        // Make host wait until all updates are complete
        _(cudaDeviceSynchronize());
        
        // This promises to be deadly slow. We'll need to figure out some
        // intelligent way of running this asynchronously. Lower frequency
        // requirement will help a lot. Benchmarking will tell us what the
        // highest possible frequency with zero lag will be.
        
        if(t % 1 == 0) {
            // Fetch from GPU
            // This memcpy _should_ be synchronous...
            _(cudaMemcpy(E_phi, data.E_phi, SIZE_NU * SIZE_MU * sizeof(float), 
                         cudaMemcpyDeviceToHost));
            
            // Write into file
            std::fstream f;
            char filename[50];
            sprintf(filename, "output/dipole/time-step-%d.txt", t);
            f.open(filename, std::fstream::out);
            f<<SIZE_NU<<std::endl<<SIZE_MU<<std::endl;
            for(int i=0 ; i < SIZE_NU ; i++) {
                for(int j=0 ; j < SIZE_MU ; j++ ) {
                    f<<E_phi[i][j]<<std::endl;
                }
            }
            f.close();
        }
        
        // Increment time step
        t++;
    }
    
    std::cout<<"All done\n";
    
    // Clean up
    _(cudaStreamDestroy(stream1));
    _(cudaStreamDestroy(stream2));
    _(cudaStreamDestroy(stream3));
    _(cudaDeviceReset());
    return 0;
}

