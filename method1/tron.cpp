#include <math.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include "tron.h"
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

#ifndef min
template <class T> static inline T min(T x,T y) { return (x<y)?x:y; }
#endif

#ifndef max
template <class T> static inline T max(T x,T y) { return (x>y)?x:y; }
#endif

#ifdef __cplusplus
extern "C" {
#endif

extern double dnrm2_(int *, double *, int *);
extern double ddot_(int *, double *, int *, double *, int *);
extern int daxpy_(int *, double *, double *, int *, double *, int *);
extern int dscal_(int *, double *, double *, int *);

#ifdef __cplusplus
}
#endif

static void default_print(const char *buf)
{
	fputs(buf,stdout);
	fflush(stdout);
}

void TRON::info(const char *fmt,...)
{
	char buf[BUFSIZ];
	va_list ap;
	va_start(ap,fmt);
	vsprintf(buf,fmt,ap);
	va_end(ap);
	(*tron_print_string)(buf);
}

TRON::TRON(const function *fun_obj, double eps, int max_cg_iter, int num_sample, int max_iter, double my_c, int solver_type)
{
	this->fun_obj=const_cast<function *>(fun_obj);
	this->eps=eps;
	this->max_iter=max_iter;
	this->max_cg_iter=max_cg_iter;
	this->num_sample=num_sample;
	this->solver_type=solver_type;
	tron_print_string = default_print;
	printf("max_newton_iter: %d, max_cg_iter: %d, num_sample: 1/%d c: %g\n",max_iter,max_cg_iter,num_sample, my_c);
}

TRON::~TRON()
{
}

void TRON::tron(double *w)
{
	int i;

	srand(0);
	double eta = 1e-4;
	
	int n = fun_obj->get_nr_variable();
	int l = fun_obj->get_nr_instance();
	double gnorm1, alpha, f, fnew, actred, gs;
	int cg_iter, search = 1, iter = 1, inc = 1;
	long int num_data = 0;
	int *Hsample = new int[l];
	double *s = new double[n];
	double *r = new double[n];
	double *w_new = new double[n];
	double *g = new double[n];

	for (i=0; i<l; i++)
		Hsample[i] = rand() % num_sample;

	for (i=0; i<n; i++)
	{
		s[i] = 0.0;
		w[i] = 0.0;
	}
	f = fun_obj->fun(w, &num_data);
	iter = 1;
	fun_obj->grad(w, g, Hsample, iter, num_sample, &num_data);
	
	gnorm1 = dnrm2_(&n, g, &inc);
	double gnorm = gnorm1;

	if (gnorm <= eps*gnorm1)
		search = 0;

	double sTs,shift_time = 0.0,cg_time = 0.0,line_search_time = 0.0;

	const clock_t start = clock();

	while (iter <= max_iter && search)
	{
		info("FUN %20.15e TIME %20.15e\n", f,(double)(clock()-start)/CLOCKS_PER_SEC);

		const clock_t begin_time = clock();

        cg_iter = trcg(g, s, r, num_sample,&num_data);
		cg_time = float(clock() - begin_time)/CLOCKS_PER_SEC;
	
        gs = ddot_(&n, g, &inc, s, &inc);

		// ***Method 1***
		if(num_sample == 1)
		{
			alpha = 1;
		}else{
			sTs = ddot_(&n, s, &inc, s, &inc);
			fun_obj->get_alpha(&alpha, gs, sTs, s, &num_data);
			for(i=0;i<n;i++)
				s[i] *= alpha;
			gs = alpha*gs;
			alpha = 1;		
		}


		const clock_t line_begin_time = clock();
		alpha = 1;
                while(1)
                {
                        memcpy(w_new, w, sizeof(double)*n);
                        daxpy_(&n, &alpha, s, &inc, w_new, &inc);
                        fnew = fun_obj->fun(w_new,&num_data);
                        actred = f - fnew;
                        if (actred+eta*alpha*gs >= 0)
                                break;
			alpha /= 2;
                }
		line_search_time = float( clock() - line_begin_time)/CLOCKS_PER_SEC;
        memcpy(w, w_new, sizeof(double)*n);
        f = fnew;

		fun_obj->grad(w, g, Hsample, iter+1, num_sample,&num_data);
		gnorm = dnrm2_(&n, g, &inc);

		shift_time += float(clock() - begin_time)/CLOCKS_PER_SEC;
		info("iter %2d act %5.3e fun %5.3e |g| %5.3e CG %3d cg_time: %g line_search_time: %g time %.6f num_data %ld f %.16f alpha %g\n", iter, actred, f, gnorm, cg_iter, cg_time, line_search_time, shift_time, num_data, f, alpha);
		iter++;

		if (gnorm <= eps*gnorm1)
			break;
	
		if(fabs(actred) <= 1.0e-12*fabs(f))
		{
			printf("actual reduction is too small");
			break;
		}
		if (f < -1.0e+32)
		{
			info("WARNING: f < -1.0e+32\n");
			break;
		}
	}

	delete[] g;
	delete[] r;
	delete[] w_new;
	delete[] s;
	delete[] Hsample;
}

int TRON::trcg(double *g, double *s, double *r, int num_sample, long int *num_data)
{
	int i, inc = 1;
	int n = fun_obj->get_nr_variable();
	double one = 1;
	double *d = new double[n];
	double *Hd = new double[n];
	double rTr, rnewTrnew, alpha, beta, cgtol;

	for (i=0; i<n; i++)
	{
		Hd[i] = 0.0;
		s[i] = 0.0;
		r[i] = -g[i];
		d[i] = r[i];
	}
	
	cgtol = 0.1*dnrm2_(&n, g, &inc);

	int cg_iter = 0;
	rTr = ddot_(&n, r, &inc, r, &inc);
	while (1)
	{
		if (dnrm2_(&n, r, &inc) <= cgtol || cg_iter >= max_cg_iter)
			break;
		cg_iter++;
		fun_obj->sample_Hv(d, Hd, num_data);

		alpha = rTr/ddot_(&n, d, &inc, Hd, &inc);
		daxpy_(&n, &alpha, d, &inc, s, &inc);
		alpha = -alpha;
		daxpy_(&n, &alpha, Hd, &inc, r, &inc);
		rnewTrnew = ddot_(&n, r, &inc, r, &inc);
		beta = rnewTrnew/rTr;
		dscal_(&n, &beta, d, &inc);
		daxpy_(&n, &one, r, &inc, d, &inc);
		rTr = rnewTrnew;
	}
	delete[] d;
	delete[] Hd;

	return(cg_iter);
}

double TRON::norm_inf(int n, double *x)
{
	double dmax = fabs(x[0]);
	for (int i=1; i<n; i++)
		if (fabs(x[i]) >= dmax)
			dmax = fabs(x[i]);
	return(dmax);
}

void TRON::set_print_string(void (*print_string) (const char *buf))
{
	tron_print_string = print_string;
}
