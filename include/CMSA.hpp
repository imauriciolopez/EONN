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
        return std::make_tuple(0, 
                               std::vector<double>{0.0},
                               std::vector<double>{0.0},
                               std::vector<double>{0.0},
                               std::vector<double>{0.0},
                               std::vector<double>{0.0},
                               std::vector<double>{0.0},
                               std::vector<double>{0.0}, 
                               std::vector<double>{0.0}, 
                               std::vector<double>{0.0},
                               std::vector<double>{0.0}, 
                               std::vector<bool>{false});
    }
}; 

template<typename T> thread_local std::mt19937 CMSA<T>::rndm{std::random_device{}()};

template<typename T> thread_local std::uniform_int_distribution<int> CMSA<T>::rndm_int{0, 10000000};

template<typename T> thread_local std::uniform_real_distribution<double> CMSA<T>::rndm_dbl{0.0, 1.0};

#endif