#include <iostream>

#ifndef ED_HPP
#define ED_HPP

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

void hello_world_ED();

template<typename T>
class ED{
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
    
    static thread_local std::mt19937 rndm;

    static thread_local std::uniform_int_distribution<int>     rndm_int;
    static thread_local std::uniform_real_distribution<double> rndm_dbl;

    ED(MLP<T> *multi_lp, 
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

        for(int iter=0;iter<n_its;iter++){

            if(verbose>0&&iter%verbose==0){
                std::cout<<"ITER: "<<iter<<std::endl;
                std::cout<<"TIEMPO TOTAL: "<<(double(clock()-t0)/CLOCKS_PER_SEC)<<std::endl;
                std::cout<<"TIEMPO DESDE LA ULTIMA ITERACION: "<<(double(clock()-t)/CLOCKS_PER_SEC)<<std::endl;
            }
            if(iter>0){
                tiempo_por_iteracion[iter-1]=(double(clock()-t)/CLOCKS_PER_SEC);
            }
            
            t=clock();
            
            std::vector<struct individuo*> n_pob(n_inds);
            
            //MUTACION
            for(int ind=0;ind<n_inds;ind++){
                //seleccionando los individuos aleatorios
                
                int rand_1=rndm_int(rndm)%n_inds;
                int rand_2=rndm_int(rndm)%n_inds;
                while(rand_1==rand_2) rand_2=rndm_int(rndm)%n_inds;
                int rand_3=rndm_int(rndm)%n_inds;
                while(rand_1==rand_3||rand_2==rand_3) rand_3=rndm_int(rndm)%n_inds;

                //haciendo la mutación
                
                struct individuo *ind_1=(*pob[rand_3])-pob[rand_2];
                struct individuo *ind_2=(*ind_1)*F;
                n_pob[ind]=(*pob[rand_1])+ind_2;
                
                //borrando los individuos que ya no se usan
                delete ind_1;
                delete ind_2;
            }
            
            //CROSSOVER
            for(int ind=0;ind<n_inds;ind++){
                n_pob[ind]->crossover(pob[ind], rndm_int(rndm)%100, C_r);
                if(json["Clip"]==1){
                    n_pob[ind]->clip(lower_bound, upper_bound);
                }
            }
            
            //EVAL
            for(int ind=0;ind<n_inds;ind++){
                n_pob[ind]->fitness=mlp->eval_train(n_pob[ind]);
                //std::cout<<n_pob[ind]->fitness<<std::endl;
            }
            
            //TORNEO
            for(int ind=0;ind<n_inds;ind++){
                struct individuo *perdedor;
                if(n_pob[ind]->fitness > pob[ind]->fitness){
                    perdedor=n_pob[ind];
                }
                else{
                    perdedor=pob[ind];
                    pob[ind]=n_pob[ind];
                }
                delete perdedor;
            }
            
            //ordenamos la poblacion
            std::vector<double> fitnesses(n_inds);
            for(int i=0;i<n_inds;i++) fitnesses[i]=pob[i]->fitness;
            std::vector<int> ordenados=b_u_merge_sort<double>(fitnesses, true, false);
            std::vector<struct individuo*> pob_copia(n_inds);

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
}; 

template<typename T> thread_local std::mt19937 ED<T>::rndm{std::random_device{}()};

template<typename T> thread_local std::uniform_int_distribution<int> ED<T>::rndm_int{0, 10000000};

template<typename T> thread_local std::uniform_real_distribution<double> ED<T>::rndm_dbl{0.0, 1.0};

#endif