#include <iostream>

#ifndef PSO_HPP
#define PSO_HPP

#include <fstream>
#include <string>

#include <type_traits>
#include <../nlohmann/json.hpp>
#include "MLP.hpp"
#include <iostream>
#include <cmath>
#include <iterator>
#include <random>
#include <vector>
#include <unordered_map>
#include <stdexcept>
#include <ctime>

#include <typeinfo>

void hello_world_PSO();

template<typename T>
class PSO{
public:
    //atributos generales para todas las clases
    double lower_bound=-1.0;
    double upper_bound=1.0;
    std::vector<struct individuo*> pob;
    int n_inds;
    int n_its;
    double F;
    double C_r;
    unsigned int seed;
    MLP<T> *mlp;

    //atributos especiales para pso
    std::vector<struct individuo*> vel;          // velocidad del individuo
    std::vector<struct individuo*> pbest;        // mejor posición personal
    std::vector<double> pbest_fit;               // mejor fitness personal
    struct individuo* gbest;                     // mejor individuo global
    double gbest_fit;                            // fitness del mejor global
    
    static thread_local std::mt19937 rndm;

    static thread_local std::uniform_int_distribution<int>     rndm_int;
    static thread_local std::uniform_real_distribution<double> rndm_dbl;
    static thread_local std::uniform_real_distribution<double> rndm_dbl_bounded;

    PSO(MLP<T> *multi_lp, 
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
        vel=std::vector<struct individuo*>(n_inds, nullptr);          // velocidad del individuo
        pbest=std::vector<struct individuo*>(n_inds, nullptr); ;        // mejor posición personal
        pbest_fit=std::vector<double>(n_inds, 100000.0);               // mejor fitness personal
        gbest=nullptr;                     // mejor individuo global
        gbest_fit=100000.0;                            // fitness del mejor global
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
        if(!vel.empty()){
            for(int i=0;i<vel.size();i++){
                delete vel[i];
                vel[i]=nullptr;
            }
        }
        if(!pbest.empty()){
            for(int i=0;i<pbest.size();i++){
                delete pbest[i];
                pbest[i]=nullptr;
            }
        }
        delete gbest;
    }

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
        
        struct individuo *desechable=new individuo(pob[0]->n_capas,        //numero de capas
                                          pob[0]->estructura,      //estructura
                                          pob[0]->lower_bound, 
                                          pob[0]->upper_bound,
                                          {{{}}},           //w
                                          {{}},           //b
                                          {},    //activaciones
                                          {},  //bias de decisión
                                          0.0,             //fitness
                                          pob[0]->seed+(unsigned int)rndm_int(rndm));//seed 
        
        for(int i = 0; i < n_inds; i++){
            std::vector<std::vector<std::vector<double> > > w_ind=desechable->w_random(pob[i]->n_capas, pob[i]->estructura);
            std::vector<std::vector<double> > b_ind=desechable->b_random(pob[i]->n_capas, pob[i]->estructura);
            std::vector<double> b_decision_ind=desechable->b_decision_random(pob[i]->n_capas, pob[i]->estructura);
            
            if(vel[i] != nullptr){
                delete vel[i];
                vel[i] = nullptr;
            }

            vel[i]=new struct individuo(pob[i]->n_capas,        //numero de capas
                                          pob[i]->estructura,      //estructura
                                          pob[i]->lower_bound,     //lower bound
                                          pob[i]->upper_bound,      //upper bound
                                          w_ind,           //w
                                          b_ind,           //b
                                          pob[i]->activaciones,    //activaciones
                                          b_decision_ind,  //bias de decisión
                                          0.0,             //fitness
                                          desechable->seed+(unsigned int)rndm_int(rndm));//seed   

            vel[i]->fitness=mlp->eval_train(vel[i]);
            
            // 5. El mejor personal inicial es el mismo individuo actual
            if(pbest[i] != nullptr){
                delete pbest[i];
                pbest[i] = nullptr;
            }

            pbest[i]=new individuo(*pob[i]);
            pbest_fit[i]=pob[i]->fitness;
            
            // 6. Actualizar mejor global
            if(pob[i]->fitness < gbest_fit){
                if(gbest!=nullptr){
                    delete gbest;
                    gbest=nullptr;
                }
                gbest = new individuo(*pob[i]);
                gbest_fit = pob[i]->fitness;
            }
        }

        delete desechable;
        
        double w_inercia  = 0.7;
        double c1 = 1.5;
        double c2 = 1.5;
        for(int iter = 0; iter < n_its; iter++){
            if(verbose>0&&iter%verbose==0){
                std::cout<<"ITER: "<<iter<<std::endl;
                std::cout<<"TIEMPO TOTAL: "<<(double(clock()-t0)/CLOCKS_PER_SEC)<<std::endl;
                std::cout<<"TIEMPO DESDE LA ULTIMA ITERACION: "<<(double(clock()-t)/CLOCKS_PER_SEC)<<std::endl;
            }
            if(iter>0){
                tiempo_por_iteracion[iter-1]=(double(clock()-t)/CLOCKS_PER_SEC);
            }
            
            t=clock();
            
            for(int i = 0; i < n_inds; i++){
                double r1 = rndm_dbl(rndm);
                double r2 = rndm_dbl(rndm);

                individuo* dif_personal = (*pbest[i]) - pob[i];
                individuo* dif_global   = (*gbest)   - pob[i];

                individuo* termino_cognitivo = (*dif_personal) * (c1 * r1);
                individuo* termino_social    = (*dif_global)   * (c2 * r2);

                individuo* velocidad_inercial = (*vel[i]) * w_inercia;

                individuo* nueva_velocidad = (*velocidad_inercial) 
                                        + termino_cognitivo;

                individuo* nueva_velocidad_2 = (*nueva_velocidad) 
                                            + termino_social;
                
                delete vel[i];
                vel[i] = nueva_velocidad_2;
                
                individuo* nueva_posicion = (*pob[i]) + vel[i];
                nueva_posicion->clip(lower_bound, upper_bound);

                
                delete pob[i];
                pob[i] = nueva_posicion;
                
                pob[i]->fitness = mlp->eval_train(pob[i]);

                if(pob[i]->fitness < pbest_fit[i]){
                    delete pbest[i];
                    pbest[i] = new individuo(*pob[i]);
                    pbest_fit[i] = pob[i]->fitness;

                    if(pbest_fit[i] < gbest_fit){
                        delete gbest;
                        gbest = new individuo(*pbest[i]);
                        gbest_fit = pbest_fit[i];
                    }
                }

                delete dif_personal;
                delete dif_global;
                delete termino_cognitivo;
                delete termino_social;
                delete velocidad_inercial;
                delete nueva_velocidad;
            }

            //ordenamos la poblacion
            std::vector<double> fitnesses(n_inds);
            for(int i=0;i<n_inds;i++) fitnesses[i]=pbest[i]->fitness;

            std::vector<int> ordenados=b_u_merge_sort<double>(fitnesses, true, false);
            std::vector<struct individuo*> pob_copia(n_inds);
            for(int i=0;i<n_inds;i++) pob_copia[i]=pob[ordenados[i]];
            pob=pob_copia;
            
            //hacemos registro de como va la población
            double best_loss_;
            
            if(json["Reporte"]=="test"){
                best_loss_=mlp->eval_test(pbest[0]);
            }
            else{
                best_loss_=mlp->eval_train(pbest[0]);
            }
            best_loss[iter]=best_loss_;
            
            double worst_loss_;
            
            if(json["Reporte"]=="test"){
                worst_loss_=mlp->eval_test(pbest[n_inds-1]);
            }
            else{
                worst_loss_=mlp->eval_train(pbest[n_inds-1]);
            }
            worst_loss[iter]=worst_loss_;

            double best_accuracy_;
            
            if(json["Reporte"]=="test"){
                best_accuracy_=mlp->eval_test_accuracy(pbest[0]);
            }
            else{
                best_accuracy_=mlp->eval_train_accuracy(pbest[0]);
            }
            best_accuracy[iter]=best_accuracy_;

            double worst_accuracy_;
            if(json["Reporte"]=="test"){
                worst_accuracy_=mlp->eval_test_accuracy(pbest[n_inds-1]);
            }
            else{
                worst_accuracy_=mlp->eval_train_accuracy(pbest[n_inds-1]);
            }
            worst_accuracy[iter]=worst_accuracy_;

            double iteracion_mejor_=pbest[0]->fitness;
            iteracion_mejor[iter]=iteracion_mejor_;
            double iteracion_peor_=pbest[n_inds-1]->fitness;
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
}; 

template<typename T> thread_local std::mt19937 PSO<T>::rndm{std::random_device{}()};

template<typename T> thread_local std::uniform_int_distribution<int> PSO<T>::rndm_int{0, 10000000};

template<typename T> thread_local std::uniform_real_distribution<double> PSO<T>::rndm_dbl{0.0, 1.0};

#endif