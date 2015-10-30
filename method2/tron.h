#ifndef _TRON_H
#define _TRON_H

class function
{
public:
	virtual double fun(double *w, long int *num_data) = 0 ;
	virtual void get_alpha(double *alpha, double *beta, double gs, double golds, double sTs, double *s, double *soldTsold, double *sold, double soldTs, int iter,long int *num_data) = 0 ;
	virtual void grad(double *w, double *g, int *Hsample, int now_iter, int num_sample, long int *num_data) = 0 ;
	virtual void Hv(double *s, double *Hs) = 0 ;
	virtual void sample_Hv(double *s, double *Hs, long int *num_data) = 0 ;

	virtual int get_nr_variable(void) = 0 ;
	virtual int get_nr_instance(void) = 0 ;
	virtual ~function(void){}
};

class TRON
{
public:
	TRON(const function *fun_obj, double eps = 0.1, int max_cg_iter = 10, int num_sample = 20, int max_iter = 1000, double my_c = 1.0, int solver_type = 0);
	~TRON();

	void tron(double *w);
	void set_print_string(void (*i_print) (const char *buf));

private:
	int trcg(double *g, double *s, double *r, double *sold, int num_sample, long int *num_data);
	double norm_inf(int n, double *x);

	double eps, my_c;
	long int count;
	int max_iter;
	int solver_type;
	int max_cg_iter;
	int num_sample;
	function *fun_obj;
	void info(const char *fmt,...);
	void (*tron_print_string)(const char *buf);
};
#endif
