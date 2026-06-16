#include <iostream>

#ifndef CMSA_HPP
#define CMSA_HPP

#include <fstream>
#include <string>

#include <type_traits>
#include <../nlohmann/json.hpp>
#include <MLP.hpp>
#include <iostream>
#include <cmath>
#include <iterator>
#include <random>
#include <vector>
#include <unordered_map>
#include <stdexcept>
#include <ctime>

#include <typeinfo>

#include <Eigen/Dense>

using namespace Eigen;

void hello_world_CMSA();

template<typename T>
class CMSA{
public:
    double lower_bound;
    double upper_bound;
    std::vector<struct individuo*> pob;
    int n_inds;
    int n_its;
    double F;
    double C_r;
    unsigned int seed;
    MLP<T> *mlp;

    bool cma_initialized = false;

    VectorXd m;
    MatrixXd C;
    MatrixXd B;
    VectorXd D;
    double sigma;
    
    static thread_local std::mt19937 rndm;

    static thread_local std::uniform_int_distribution<int>     rndm_int;
    static thread_local std::uniform_real_distribution<double> rndm_dbl;

    CMSA(MLP<T> *multi_lp, 
       unsigned int seed, 
       double lower_bound, 
       double upper_bound,
       int n_inds, 
       int n_its, 
       double F, 
       double C_r):
       upper_bound(upper_bound), 
       lower_bound(lower_bound), 
       seed(seed),
       mlp(multi_lp),
       n_inds(n_inds), 
       n_its(n_its), 
       F(F), 
       C_r(C_r){
        rndm.seed(seed);
       }

    void ver(){
        std::cout<<"MLP: "<<std::endl;
        mlp->ver();
        std::cout<<"seed: "<<seed<<std::endl;
        std::cout<<"lower_bound: "<<lower_bound<<std::endl;
        std::cout<<"upper_bound: "<<upper_bound<<std::endl;
        std::cout<<"n_inds: "<<n_inds<<std::endl;
        std::cout<<"n_its: "<<n_its<<std::endl;
        std::cout<<"F: "<<F<<std::endl;
        std::cout<<"C_r: "<<C_r<<std::endl;
    }

    void inicializar_poblacion(int n_capas, std::vector<int> estructura, std::vector<double(*)(double, double)> activaciones){
        pob=std::vector<struct individuo*>(n_inds);
        struct individuo *desechable=new individuo(n_capas,        //numero de capas
                                          estructura,      //estructura
                                          lower_bound, 
                                          upper_bound,
                                          {{{}}},           //w
                                          {{}},           //b
                                          {},    //activaciones
                                          {},  //bias de decisión
                                          0.0,             //fitness
                                          seed+(unsigned int)rndm_int(rndm));//seed 

        for(int ind=0;ind<n_inds;ind++){
            
            std::vector<std::vector<std::vector<double> > > w_ind=desechable->w_random(n_capas, estructura);
            std::vector<std::vector<double> > b_ind=desechable->b_random(n_capas, estructura);
            std::vector<double> b_decision_ind=desechable->b_decision_random(n_capas, estructura);
            
            pob[ind]=new struct individuo(n_capas,        //numero de capas
                                          estructura,      //estructura
                                          lower_bound,     //lower bound
                                          upper_bound,      //upper bound
                                          w_ind,           //w
                                          b_ind,           //b
                                          activaciones,    //activaciones
                                          b_decision_ind,  //bias de decisión
                                          0.0,             //fitness
                                          desechable->seed+(unsigned int)rndm_int(rndm));//seed   

            pob[ind]->fitness=mlp->eval_train(pob[ind]);

            pob[ind]->n=pob[ind]->aplanar_parametros().size();
        }
        delete desechable;
    }

    void liberar_poblacion(){
        if(!pob.empty()){
            for(int i=0;i<pob.size();i++){
                delete pob[i];
                pob[i]=nullptr;
            }
        }
    }

    /*
    void actualizar_matriz_covarianza(
        Eigen::MatrixXd& C,
        const std::vector<double>& p_c,
        const std::vector<std::vector<double>>& y_pop,
        const std::vector<std::pair<double, int>>& aptitudes,
        const std::vector<double>& w,
        double c_1,
        double c_mu,
        double h_sigma,
        double c_c
    ) {
        int n = C.rows();
        int mu = w.size();

        // 1. Convertir p_c (std::vector) a un VectorXd de Eigen
        Eigen::Map<const Eigen::VectorXd> p_c_eigen(p_c.data(), n);

        // 2. Calcular el término de Rango-1: p_c * p_c^T
        Eigen::MatrixXd rango_1 = p_c_eigen * p_c_eigen.transpose();

        // 3. Calcular el término de Rango-μ: sum( w_i * y_i * y_i^T )
        Eigen::MatrixXd rango_mu = Eigen::MatrixXd::Zero(n, n);
        for (int i = 0; i < mu; ++i) {
            int idx = aptitudes[i].second; // Índice del i-ésimo mejor individuo
            
            // Mapear el vector de desviaciones y_k del individuo seleccionado
            Eigen::Map<const Eigen::VectorXd> y_i(y_pop[idx].data(), n);
            
            // Acumular la matriz externa ponderada por su peso w_i
            rango_mu += w[i] * (y_i * y_i.transpose());
        }

        // 4. Factor de ajuste histórico si h_sigma es cero (ocurre cuando el paso crece muy rápido)
        double ajuste_h = (1.0 - h_sigma) * c_c * (2.0 - c_c);

        // 5. Aplicar la ecuación de actualización completa sobre la matriz C
        C = (1.0 - c_1 - c_mu) * C + 
            c_1 * (rango_1 + ajuste_h * C) + 
            c_mu * rango_mu;

        // 6. Regularización: Forzar simetría estricta para mitigar errores numéricos de flotantes
        C = 0.5 * (C + C.transpose());
    }*/
   
    

    std::tuple<unsigned int,
               std::vector<double>, 
               std::vector<double>, 
               std::vector<double>, 
               std::vector<double>, 
               std::vector<double>, 
               std::vector<double>,
               std::vector<double>, 
               std::vector<double>,
               std::vector<double>, 
               std::vector<double>, 
               std::vector<bool> > optimize(int verbose, nlohmann::json json){



        unsigned t0=clock();
        unsigned t;

        //parámetros de avance
        std::vector<double> best_loss(n_its);
        std::vector<double> worst_loss(n_its);

        std::vector<double> best_accuracy(n_its);
        std::vector<double> worst_accuracy(n_its);

        std::vector<double> iteracion_mejor(n_its);
        std::vector<double> iteracion_peor(n_its);
        std::vector<double> promedio(n_its);
        std::vector<double> desvest(n_its);

        std::vector<double> tiempo_por_iteracion(n_its);

        std::vector<bool> succes(n_its);

        /*
        int n=pob[0]->aplanar_parametros().size();
        std::vector<double> x_mean=pob[0]->aplanar_parametros();

        //inicializando los parámetros
        int mu=n_inds/2;
        std::vector<double> w(mu);

        double sum_w=0.0, sum_w_sq=0.0;
        for (int i=0;i<mu;i++){
            w[i]=std::log(mu+0.5)-std::log(i+1.0);
            sum_w+=w[i];
        }
        for (int i=0;i<mu;i++){
            w[i]/=sum_w;
            sum_w_sq+=w[i]*w[i];
        }
        double mu_w=1.0/sum_w_sq;

        //parámetros de aprendizaje
        double c_sigma=(mu_w+2.0)/(n+mu_w+5.0);
        double d_sigma=1.0+2.0*std::max(0.0, std::sqrt((mu_w-1.0)/(n+1.0))-1.0)+c_sigma;
        double c_c=4.0/(n+4.0);
        double c_1=2.0/(std::pow(n+1.3,2)+mu_w);
        double c_mu=std::min(1.0-c_1,2.0*(mu_w-2.0+1.0/mu_w)/(std::pow(n+2.0,2)+mu_w));
        double e_chi=std::sqrt(n)*(1.0-1.0/(4.0*n)+1.0/(21.0*n*n));

        //estado inicial
        std::vector<double> p_c(n, 0.0), p_sigma(n, 0.0);
        //representación de la matriz de covarianza
        std::vector<std::vector<double> > C=Eigen::MatrixXd C(n, n);

        std::normal_distribution<double> d_norm(0.0, 1.0);
        */
        
        std::vector<double> flat=pob[0]->aplanar_parametros();

        int dim=flat.size();
        int n_padres=json["N padres"].get<int>();

        VectorXd weights(n_padres);
        double sum_w=0.0, sum_w_sq=0.0;
        for(int i=0;i<n_padres;i++){
            weights(i)=log(n_padres+0.5)-log(i+1.0);
            sum_w+=weights(i);
        }
        for(int i=0;i<n_padres;i++){
            weights(i)/=sum_w;
            sum_w_sq+=weights(i)*weights(i);
        }
        double mu_w=1.0/sum_w_sq;

        
        double c_sigma=(mu_w+2.0)/(dim + mu_w+5.0);
        double d_sigma=1.0+2.0*std::max(0.0, sqrt((mu_w-1.0)/(dim+1.0))-1.0)+c_sigma;
        double c_c=4.0/(dim+4.0);
        double c_1=2.0/(pow(dim+1.3, 2)+mu_w);
        double c_mu=std::min(1.0-c_1, 2.0*(mu_w-2.0+1.0/mu_w)/(pow(dim+2.0, 2)+mu_w));
        double E_chi=sqrt(dim)*(1.0-1.0/(4.0*dim)+1.0/(21.0*dim*dim));

        VectorXd p_c=VectorXd::Zero(dim);
        VectorXd p_sigma=VectorXd::Zero(dim);

        double sigma=0.5; 

        /*
        for(int iter=0;iter<n_its;iter++){
            double c_cov=0.2;

            //Inicialización en la primera iteración
            if(!cma_initialized){
                sigma=0.5;
                //Media inicial = promedio población
                m=VectorXd::Zero(dim);

                for(int i=0;i<n_inds;i++){
                    std::vector<double> x_flat=pob[i]->aplanar_parametros();
                    for(int j=0;j<dim;j++) m(j)+=x_flat[j];
                }

                m/=n_inds;

                //Covarianza inicial
                C=MatrixXd::Identity(dim, dim);

                //Descomposición inicial
                SelfAdjointEigenSolver<MatrixXd>eig(C);
                B=eig.eigenvectors();
                D=eig.eigenvalues();

                for(int i=0;i<dim;i++) D(i)=sqrt(std::max(1e-20, D(i)));

                cma_initialized=true;
            }

            //Recombinación de pesos
            MatrixXd BD=B*D.asDiagonal();
            std::normal_distribution<double> gauss(0.0, 1.0);

            std::vector<VectorXd> x_pop(n_inds, VectorXd(dim));
            std::vector<VectorXd> y_pop(n_inds, VectorXd(dim));

            // Muestreo y Evaluación
            for(int i=0;i<n_inds;i++){
                VectorXd z(dim);
                for(int j=0;j<dim;j++) z(j)=gauss(rndm);
                
                y_pop[i]=BD*z;
                x_pop[i]=m+sigma*y_pop[i];

                std::vector<double> nuevo(dim);
                for(int j=0;j<dim;j++){
                    // Aplicar límites
                    nuevo[j] = std::max(lower_bound, std::min(upper_bound, x_pop[i](j)));
                }

                pob[i]->desaplanar_parametros(nuevo);
                pob[i]->fitness = mlp->eval_train(pob[i]);
            }

            //ordenamos la poblacion
            std::vector<double> fitnesses(n_inds);
            for(int i=0;i<n_inds;i++) fitnesses[i]=pob[i]->fitness;
            std::vector<int> ordenados=b_u_merge_sort<double>(fitnesses, true, false);
            std::vector<struct individuo*> pob_copia(n_inds);

            for(int i=0;i<n_inds;i++) pob_copia[i]=pob[ordenados[i]];
            pob=pob_copia;

            //Guardar media anterior
            VectorXd old_m=m;

            //Nueva media
            m.setZero();

            for(int i=0;i<json["N padres"];i++){
                std::vector<double> x_flat=pob[i]->aplanar_parametros();

                for(int j=0;j<dim;j++) m(j)+=weights(i)*x_flat[j];
            }

            //Adaptación covarianza
            MatrixXd rank_mu=MatrixXd::Zero(dim, dim);

            for(int i=0;i<json["N padres"];i++){
                std::vector<double> x_flat=pob[i]->aplanar_parametros();

                VectorXd y(dim);

                for(int j=0;j<dim;j++) y(j)=(x_flat[j]-old_m(j))/sigma;

                rank_mu+=weights(i)*y*y.transpose();
            }

            C=(1.0-c_cov)*C+c_cov*rank_mu;

            //Recalcular eigensystem
            SelfAdjointEigenSolver<MatrixXd> eig(C);

            B=eig.eigenvectors();

            D=eig.eigenvalues();

            for(int i=0;i<dim;i++) D(i)=sqrt(std::max(1e-20, D(i)));
    
            //ordenamos la poblacion
            for(int i=0;i<n_inds;i++) fitnesses[i]=pob[i]->fitness;
            ordenados=b_u_merge_sort<double>(fitnesses, true, false);

            for(int i=0;i<n_inds;i++) pob_copia[i]=pob[ordenados[i]];
            pob=pob_copia;
            
            //hacemos registro de como va la población
            double best_loss_;
            if(json["Reporte"]=="test"){
                best_loss_=mlp->eval_test(pob[0]);
            }
            else{
                best_loss_=mlp->eval_train(pob[0]);
            }
            best_loss[iter]=best_loss_;
            
            double worst_loss_;
            if(json["Reporte"]=="test"){
                worst_loss_=mlp->eval_test(pob[n_inds-1]);
            }
            else{
                worst_loss_=mlp->eval_train(pob[n_inds-1]);
            }
            worst_loss[iter]=worst_loss_;

            double best_accuracy_;
            if(json["Reporte"]=="test"){
                best_accuracy_=mlp->eval_test_accuracy(pob[0]);
            }
            else{
                best_accuracy_=mlp->eval_train_accuracy(pob[0]);
            }
            best_accuracy[iter]=best_accuracy_;

            double worst_accuracy_;
            if(json["Reporte"]=="test"){
                worst_accuracy_=mlp->eval_test_accuracy(pob[n_inds-1]);
            }
            else{
                worst_accuracy_=mlp->eval_train_accuracy(pob[n_inds-1]);
            }
            worst_accuracy[iter]=worst_accuracy_;

            double iteracion_mejor_=pob[0]->fitness;
            iteracion_mejor[iter]=iteracion_mejor_;
            double iteracion_peor_=pob[n_inds-1]->fitness;
            iteracion_peor[iter]=iteracion_peor_;
            std::vector<double> promedio_desvest=mean_desv_est<double>(fitnesses);
            double promedio_=promedio_desvest[0];
            promedio[iter]=promedio_;
            double desvest_=promedio_desvest[1];
            desvest[iter]=desvest_;

            bool succes_=best_accuracy_==1.0;
            succes[iter]=succes_;
            
            if(verbose>0&&iter%verbose==0){
                std::cout<<"best_loss: "<<best_loss_<<std::endl;
                std::cout<<"worst_loss: "<<worst_loss_<<std::endl;

                std::cout<<"best_accuracy: "<<best_accuracy_<<std::endl;
                std::cout<<"worst_accuracy: "<<worst_accuracy_<<std::endl;

                std::cout<<"iteracion_mejor: "<<iteracion_mejor_<<std::endl;
                std::cout<<"iteracion_peor: "<<iteracion_peor_<<std::endl;

                std::cout<<"promedio: "<<promedio_<<std::endl;
                std::cout<<"desves: "<<desvest_<<std::endl;

                std::cout<<"succes: "<<succes_<<std::endl;
            }
        }
        */

        // Antes del bucle for(int iter=0...)
        double mejor_fitness_global = std::numeric_limits<double>::max(); // Suponiendo que minimizas
        std::vector<double> mejores_parametros_globales;
        double mejor_fitness_historico = std::numeric_limits<double>::max();
        VectorXd mejor_x_historico = VectorXd::Zero(dim);

        for(int iter=0;iter<n_its;iter++){
            // Inicialización diferida (C, m, B, D) se mantiene igual a tu código original
            if(!cma_initialized){
                m = VectorXd::Zero(dim);
                for(int i = 0; i < n_inds; i++){
                    std::vector<double> x_flat = pob[i]->aplanar_parametros();
                    for(int j = 0; j < dim; j++) m(j) += x_flat[j];
                }
                m /= n_inds;
                C = MatrixXd::Identity(dim, dim);
                SelfAdjointEigenSolver<MatrixXd> eig(C);
                B = eig.eigenvectors();
                D = eig.eigenvalues();
                for(int i = 0; i < dim; i++) D(i) = sqrt(std::max(1e-20, D(i)));
                cma_initialized = true;
            }

            MatrixXd BD = B * D.asDiagonal();
            std::normal_distribution<double> gauss(0.0, 1.0);

            std::vector<VectorXd> x_pop(n_inds, VectorXd(dim));
            std::vector<VectorXd> y_pop(n_inds, VectorXd(dim));

            // Muestreo y Evaluación
            
            for(int i = 0; i < n_inds; i++){
                VectorXd z(dim);
                for(int j = 0; j < dim; j++) z(j) = gauss(rndm);
                
                y_pop[i] = BD * z;
                x_pop[i] = m + sigma * y_pop[i];

                std::vector<double> nuevo(dim);
                for(int j = 0; j < dim; j++){
                    // Aplicar límites SOLO para evaluar en el MLP, preservando x_pop sin alterar
                    nuevo[j] = std::max(lower_bound, std::min(upper_bound, x_pop[i](j)));
                }

                pob[i]->desaplanar_parametros(nuevo);
                pob[i]->fitness = mlp->eval_train(pob[i]); // Asignación explícita del fitness
            }
            
            /*
            for(int i = 0; i < n_inds; i++){
                if (i == 0 && iter > 0) {
                    // Inyectar el mejor histórico en la primera posición
                    x_pop[i] = mejor_x_historico; 
                    y_pop[i] = (x_pop[i] - m) / sigma; // Recalcular 'y' para mantener consistencia matemática
                    
                    std::vector<double> nuevo(dim);
                    for(int j = 0; j < dim; j++) nuevo[j] = x_pop[i](j);
                    
                    pob[i]->desaplanar_parametros(nuevo);
                    pob[i]->fitness = mejor_fitness_historico; 
                } 
                else {
                    // Muestreo normal para el resto
                    VectorXd z(dim);
                    for(int j = 0; j < dim; j++) z(j) = gauss(rndm);
                    
                    y_pop[i] = BD * z;
                    x_pop[i] = m + sigma * y_pop[i];

                    std::vector<double> nuevo(dim);
                    for(int j = 0; j < dim; j++){
                        nuevo[j] = std::max(lower_bound, std::min(upper_bound, x_pop[i](j)));
                    }

                    pob[i]->desaplanar_parametros(nuevo);
                    pob[i]->fitness = mlp->eval_train(pob[i]);
                }
            }
            if (pob[0]->fitness < mejor_fitness_global) { // pob[0] es el mejor de esta generación
                mejor_fitness_global = pob[0]->fitness;
                mejores_parametros_globales = pob[0]->aplanar_parametros();
            }*/


            // Ordenamiento sincronizado (incluye x_pop y y_pop)
            std::vector<double> fitnesses(n_inds);
            for(int i = 0; i < n_inds; i++) fitnesses[i] = pob[i]->fitness;
            std::vector<int> ordenados = b_u_merge_sort<double>(fitnesses, true, false);

            std::vector<struct individuo*> pob_copia(n_inds);
            std::vector<VectorXd> x_pop_copia(n_inds);
            std::vector<VectorXd> y_pop_copia(n_inds);

            for(int i = 0; i < n_inds; i++){
                pob_copia[i] = pob[ordenados[i]];
                x_pop_copia[i] = x_pop[ordenados[i]];
                y_pop_copia[i] = y_pop[ordenados[i]];
            }
            pob = pob_copia;
            x_pop = x_pop_copia;
            y_pop = y_pop_copia;

            // Nueva media
            VectorXd old_m = m;
            m.setZero();
            VectorXd y_w = VectorXd::Zero(dim);

            for(int i = 0; i < n_padres; i++){
                m += weights(i) * x_pop[i];
                y_w += weights(i) * y_pop[i];
            }

            // Adaptación de caminos (p_sigma y p_c)
            VectorXd z_mean = B.transpose() * y_w;
            p_sigma = (1.0 - c_sigma) * p_sigma + sqrt(c_sigma * (2.0 - c_sigma) * mu_w) * z_mean;

            double norm_p_sigma = p_sigma.norm();
            double h_sigma = (norm_p_sigma / sqrt(1.0 - pow(1.0 - c_sigma, 2 * (iter + 1))) < (1.4 + 2.0 / (dim + 1.0)) * E_chi) ? 1.0 : 0.0;

            p_c = (1.0 - c_c) * p_c + h_sigma * sqrt(c_c * (2.0 - c_c) * mu_w) * y_w;

            // Adaptación de matriz de covarianza (Rango-1 y Rango-mu)
            MatrixXd rank_1 = p_c * p_c.transpose();
            MatrixXd rank_mu = MatrixXd::Zero(dim, dim);

            for(int i = 0; i < n_padres; i++){
                rank_mu += weights(i) * (y_pop[i] * y_pop[i].transpose());
            }

            double adj_h = (1.0 - h_sigma) * c_c * (2.0 - c_c);
            C = (1.0 - c_1 - c_mu) * C + c_1 * (rank_1 + adj_h * C) + c_mu * rank_mu;

            // Forzar simetría y recalcular eigen-descomposición
            C = 0.5 * (C + C.transpose());

            SelfAdjointEigenSolver<MatrixXd> eig(C);
            B = eig.eigenvectors();
            D = eig.eigenvalues();
            for(int i = 0; i < dim; i++) D(i) = sqrt(std::max(1e-20, D(i)));

            // Actualización del tamaño de paso (Sigma)
            sigma = sigma * exp((c_sigma / d_sigma) * (norm_p_sigma / E_chi - 1.0));

            // NUEVO: Prevenir el colapso numérico
            if (sigma < 1e-8) sigma = 1e-8;
            if (sigma > 1e8) sigma = 1e8;

            //ordenamos la poblacion
            for(int i=0;i<n_inds;i++) fitnesses[i]=pob[i]->fitness;
            ordenados=b_u_merge_sort<double>(fitnesses, true, false);

            for(int i=0;i<n_inds;i++) pob_copia[i]=pob[ordenados[i]];
            pob=pob_copia;
            
            //hacemos registro de como va la población
            double best_loss_;
            if(json["Reporte"]=="test"){
                best_loss_=mlp->eval_test(pob[0]);
            }
            else{
                best_loss_=mlp->eval_train(pob[0]);
            }
            best_loss[iter]=best_loss_;
            
            double worst_loss_;
            if(json["Reporte"]=="test"){
                worst_loss_=mlp->eval_test(pob[n_inds-1]);
            }
            else{
                worst_loss_=mlp->eval_train(pob[n_inds-1]);
            }
            worst_loss[iter]=worst_loss_;

            double best_accuracy_;
            if(json["Reporte"]=="test"){
                best_accuracy_=mlp->eval_test_accuracy(pob[0]);
            }
            else{
                best_accuracy_=mlp->eval_train_accuracy(pob[0]);
            }
            best_accuracy[iter]=best_accuracy_;

            double worst_accuracy_;
            if(json["Reporte"]=="test"){
                worst_accuracy_=mlp->eval_test_accuracy(pob[n_inds-1]);
            }
            else{
                worst_accuracy_=mlp->eval_train_accuracy(pob[n_inds-1]);
            }
            worst_accuracy[iter]=worst_accuracy_;

            double iteracion_mejor_=pob[0]->fitness;
            iteracion_mejor[iter]=iteracion_mejor_;
            double iteracion_peor_=pob[n_inds-1]->fitness;
            iteracion_peor[iter]=iteracion_peor_;
            std::vector<double> promedio_desvest=mean_desv_est<double>(fitnesses);
            double promedio_=promedio_desvest[0];
            promedio[iter]=promedio_;
            double desvest_=promedio_desvest[1];
            desvest[iter]=desvest_;

            bool succes_=best_accuracy_==1.0;
            succes[iter]=succes_;
            
            if(verbose>0&&iter%verbose==0){
                std::cout<<"best_loss: "<<best_loss_<<std::endl;
                std::cout<<"worst_loss: "<<worst_loss_<<std::endl;

                std::cout<<"best_accuracy: "<<best_accuracy_<<std::endl;
                std::cout<<"worst_accuracy: "<<worst_accuracy_<<std::endl;

                std::cout<<"iteracion_mejor: "<<iteracion_mejor_<<std::endl;
                std::cout<<"iteracion_peor: "<<iteracion_peor_<<std::endl;

                std::cout<<"promedio: "<<promedio_<<std::endl;
                std::cout<<"desves: "<<desvest_<<std::endl;

                std::cout<<"succes: "<<succes_<<std::endl;
            }
        }


        tiempo_por_iteracion[n_its-1]=(double(clock()-t)/CLOCKS_PER_SEC);

        std::vector<double> pob_final(n_inds);
        for(int i=0;i<n_inds;i++){
            pob_final[i]=static_cast<double>(pob[i]->fitness);
        }
        return std::make_tuple(seed, 
                               best_loss,
                               worst_loss,
                               best_accuracy,
                               worst_accuracy,
                               iteracion_mejor,
                               iteracion_peor,
                               promedio, 
                               desvest, 
                               tiempo_por_iteracion,
                               pob_final, 
                               succes);
    }
    
    
    /*
    void CMSA::iteracion_CMA(){
        //tamaño del problema
        std::vector<double> flat=pob[0]->aplanar_parametros();
        int dim = flat.size();

        //parámetros para CMA
        int lambda=n_inds;
        int mu=lambda/2;

        double sigma=0.5;

        // Pesos recombinación
        VectorXd weights(mu);

        for(int i=0;i<mu;i++) weights(i)=log(mu+0.5)-log(i+1);

        weights/=weights.sum();

        //Construir media inicial m
        VectorXd m=VectorXd::Zero(dim);

        for(int i=0;i<lambda;i++){
            std::vector<double> x_flat=pob[i]->aplanar_parametros();
            for(int j=0;j<dim;j++) m(j)+=x_flat[j];
        }

        m/=lambda;

        //Inicializar covarianza
        MatrixXd C=MatrixXd::Identity(dim, dim);

        //Eigen decomposición
        SelfAdjointEigenSolver<MatrixXd> eig(C);

        MatrixXd B=eig.eigenvectors();
        VectorXd D=eig.eigenvalues();

        for(int i=0;i<dim;i++) D(i)=sqrt(std::max(1e-20, D(i)));

        MatrixXd BD=B*D.asDiagonal();

        //Generar nueva población
        std::normal_distribution<double> gauss(0.0, 1.0);

        for(int i=0;i<lambda;i++){
            VectorXd z(dim);

            for(int j=0;j<dim;j++) z(j)=gauss(rndm);

            VectorXd x=m+sigma*(BD*z);

            //convertir Eigen
            std::vector<double> nuevo(dim);

            for(int j=0;j<dim;j++){
                nuevo[j]=x(j);

                //haciendo clip
                nuevo[j]=std::max(lower_bound, std::min(upper_bound, nuevo[j]));
            }

            pob[i]->desaplanar_parametros(nuevo);

            // Evaluación MLP
            mlp->eval_train(*pob[i]);
        }

        //ordenamos la poblacion
        std::vector<double> fitnesses(n_inds);
        for(int i=0;i<n_inds;i++) fitnesses[i]=pob[i]->fitness;
        std::vector<int> ordenados=b_u_merge_sort<double>(fitnesses, true, false);
        std::vector<struct individuo*> pob_copia(n_inds);

        for(int i=0;i<n_inds;i++) pob_copia[i]=pob[ordenados[i]];
        pob=pob_copia;

        //Actualizar media
        VectorXd old_m=m;
        m.setZero();

        for(int i=0;i<mu;i++){
            std::vector<double> x_flat=pob[i]->aplanar_parametros();

            for(int j=0;j<dim;j++) m(j)+=weights(i)*x_flat[j];
        }

        //Actualizar covarianza
        double c_cov=0.2;

        MatrixXd rank_mu=MatrixXd::Zero(dim, dim);

        for(int i=0;i<mu;i++){
            std::vector<double> x_flat=pob[i]->aplanar_parametros();

            VectorXd y(dim);

            for(int j=0;j<dim;j++) y(j)=(x_flat[j]-old_m(j))/sigma;

            rank_mu+=weights(i)*(y*y.transpose());
        }

        C=(1.0-c_cov)*C+c_cov*rank_mu;
        }
    */


}; 

template<typename T> thread_local std::mt19937 CMSA<T>::rndm{std::random_device{}()};

template<typename T> thread_local std::uniform_int_distribution<int> CMSA<T>::rndm_int{0, 10000000};

template<typename T> thread_local std::uniform_real_distribution<double> CMSA<T>::rndm_dbl{0.0, 1.0};

#endif