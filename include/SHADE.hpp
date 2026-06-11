#include <iostream>

#ifndef SHADE_HPP
#define SHADE_HPP

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

void hello_world_SHADE();

template<typename T>
class SHADE{
public:
    //datos generales
    double lower_bound;
    double upper_bound;
    std::vector<struct individuo*> pob;
    int n_inds;
    int n_its;
    double F;
    double C_r;
    unsigned int seed;
    MLP<T> *mlp;

    //datos específicos para SHADE
    std::vector<double> mem_cr;
    std::vector<double> mem_f;
    int indice_circular;
    int tamano_memoria;
    double prom_media_aritmetica_cr;
    double prom_media_lehmer_f;
    std::vector<struct individuo*> archivo;
    
    static thread_local std::mt19937 rndm;

    static thread_local std::uniform_int_distribution<int>     rndm_int;
    static thread_local std::uniform_real_distribution<double> rndm_dbl;

    SHADE(MLP<T> *multi_lp, 
       unsigned int seed, 
       double lower_bound, 
       double upper_bound,
       int n_inds, 
       int n_its, 
       double F, 
       double C_r):
       tamano_memoria(n_inds), 
       upper_bound(upper_bound), 
       lower_bound(lower_bound),
       seed(seed),
       mlp(multi_lp),
       n_inds(n_inds), 
       n_its(n_its), 
       F(F), 
       C_r(C_r){
        rndm.seed(seed);
        mem_cr=std::vector<double>(n_inds, 0.5);
        mem_f=std::vector<double>(n_inds, 0.5);
        indice_circular=0;
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
                                          desechable->seed+ind);//seed   

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
        if(!archivo.empty()){
            for(int i=0;i<archivo.size();i++){
                delete archivo[i];
                archivo[i]=nullptr;
            }
        }
    }

    double sample_cr(double mean_cr){
        std::normal_distribution<double> norm_dist(mean_cr, 0.1);
        double answ;
        do{
            answ=mean_cr+0.1*norm_dist(rndm);
        }while(answ<0.0);
        if(answ>1.0) answ=1.0;
        return answ;
    }

    double sample_f(double mean_f){
        std::cauchy_distribution<double> cauchy_dist(0.0, 1.0);
        double answ;
        do{
            answ=mean_f+0.1*cauchy_dist(rndm);
        }while(answ<0.0);
        if(answ>1.0) answ=1.0;
        return answ;
    }

    double sample_p(){
        double p_min=2.0/(n_inds*1.0);
        std::uniform_real_distribution<double> p_dist(p_min, 0.2);
        return p_dist(rndm);
    }

    unsigned int select_pbest_index(double p){
        unsigned int p_num=(2>(unsigned int)(std::ceil(p*n_inds)))?(2):((unsigned int)(std::ceil(p*n_inds)));
        return rndm_int(rndm)%p_num;
    }

    struct individuo *mutation_current_to_pbest_1(int ind, double F, unsigned int p){
        //x_r2->puede pertenecer al archivo
        //v_i = x_i + F*(x_pbest - x_i) + F*(x_r1 - x_r2)
        //            |               |  └--------------┘pt 2
        //            └---------------┘pt 1
        int rand_1=rndm_int(rndm)%n_inds;
        int rand_2=rndm_int(rndm)%n_inds+archivo.size();
        while(rand_1==rand_2) rand_2=rndm_int(rndm)%n_inds+archivo.size();
        
        //haciendo la mutación
        struct individuo *ind_1;
        struct individuo *ind_2;
        if(rand_2<n_inds){
            ind_1=(*pob[rand_1])-pob[rand_2];
            ind_2=(*ind_1)*F;//pt 2
        }
        else{
            ind_1=(*pob[rand_1])-archivo[rand_2];
            ind_2=(*ind_1)*F;//pt 2
        }
        
        //haciendo la mutación
        struct individuo *ind_3=(*pob[p])-pob[ind];
        struct individuo *ind_4=(*ind_3)*F;//pt 1

        struct individuo *ind_5=(*pob[ind])+ind_4;//x_i + F*(x_pbest - x_i)
        struct individuo *ind_6=(*ind_5)+ind_2;   //x_i + F*(x_pbest - x_i) + F*(x_r1 - x_r2)
        
        //borrando los individuos que ya no se usan
        delete ind_1;
        delete ind_2;
        delete ind_3;
        delete ind_4;
        delete ind_5;

        return ind_6;
    }

    struct individuo *mutation_rand_1(int ind, double F, unsigned int p){
        //v_i = x_r1 + F*(x_r2 - x_r3)
        //            |              |
        //            └--------------┘pt 1
        int rand_1=rndm_int(rndm)%n_inds;
        int rand_2=rndm_int(rndm)%n_inds;
        while(rand_1==rand_2) rand_2=rndm_int(rndm)%n_inds;
        int rand_3=rndm_int(rndm)%n_inds;
        while(rand_1==rand_3||rand_3==rand_2) rand_3=rndm_int(rndm)%n_inds;
        
        //haciendo la mutación
        struct individuo *ind_1=(*pob[rand_2])-pob[rand_3];
        struct individuo *ind_2=(*ind_1)*F;//pt 1
        struct individuo *ind_3=(*pob[rand_1])+ind_2;//pt 2x_r1 + F*(x_r2 - x_r3)

        //borrando los individuos que ya no se usan
        delete ind_1;
        delete ind_2;

        return ind_3;
    }

    struct individuo *mutation_rand_2(int ind, double F, unsigned int p){
        //v_i = x_r1 + F*(x_r2 - x_r3) + F*(x_r4 - x_r5)
        //            |              |   └-------------┘pt 2
        //            └--------------┘pt 1
        int rand_1=rndm_int(rndm)%n_inds;
        int rand_2=rndm_int(rndm)%n_inds;
        while(rand_1==rand_2) rand_2=rndm_int(rndm)%n_inds;
        int rand_3=rndm_int(rndm)%n_inds;
        while(rand_1==rand_3||rand_2==rand_3) rand_3=rndm_int(rndm)%n_inds;
        int rand_4=rndm_int(rndm)%n_inds;
        while(rand_1==rand_4||rand_2==rand_4||rand_3==rand_4) rand_4=rndm_int(rndm)%n_inds;
        int rand_5=rndm_int(rndm)%n_inds;
        while(rand_1==rand_5||rand_2==rand_5||rand_3==rand_5||rand_4==rand_5) rand_5=rndm_int(rndm)%n_inds;
        
        //haciendo la mutación
        struct individuo *ind_1=(*pob[rand_2])-pob[rand_3];
        struct individuo *ind_2=(*ind_1)*F;//pt 2

        //haciendo la mutación
        struct individuo *ind_3=(*pob[rand_4])-pob[rand_5];
        struct individuo *ind_4=(*ind_3)*F;//pt 1

        struct individuo *ind_5=(*pob[rand_1])+ind_2;//x_i + F*(x_pbest - x_i)
        struct individuo *ind_6=(*ind_5)+ind_4;      //x_i + F*(x_pbest - x_i) + F*(x_r1 - x_r2)
        
        //borrando los individuos que ya no se usan
        delete ind_1;
        delete ind_2;
        delete ind_3;
        delete ind_4;
        delete ind_5;

        return ind_6;
    }

    struct individuo *mutation_best_1(int ind, double F, unsigned int p){
        //v_i = x_best + F*(x_r1 - x_r2)
        //              |              |
        //              └--------------┘pt 1
        int rand_1=rndm_int(rndm)%n_inds;
        int rand_2=rndm_int(rndm)%n_inds;
        while(rand_1==rand_2) rand_2=rndm_int(rndm)%n_inds;
        
        //haciendo la mutación
        struct individuo *ind_1=(*pob[rand_1])-pob[rand_2];
        struct individuo *ind_2=(*ind_1)*F;//pt 1

        //haciendo la mutación
        struct individuo *ind_3=(*pob[0])+ind_2;
        //borrando los individuos que ya no se usan
        delete ind_1;
        delete ind_2;

        return ind_3;
    }

    double weighted_arithmetic_mean(std::vector<double> delta_f, std::vector<double> s_cr){
        double suma=0.0;
        for(int i=0;i<delta_f.size();i++) suma+=delta_f[i];
        for(int i=0;i<delta_f.size();i++) delta_f[i]/=suma;

        suma=0.0;
        for(int i=0;i<delta_f.size();i++) suma+=s_cr[i]*delta_f[i];

        return suma;
    }

    double weighted_lehmer_mean(std::vector<double> delta_f, std::vector<double> s_f){
        double suma=0.0, numerador=0.0, denominador=0.0;

        for(int i=0;i<delta_f.size();i++) suma+=delta_f[i];
        for(int i=0;i<delta_f.size();i++) delta_f[i]/=suma;

        for(int i=0;i<delta_f.size();i++){
            numerador+=delta_f[i]*std::pow(s_f[i], 2.0);
            denominador+=s_f[i]*delta_f[i];
        }
        if(denominador==0.0){
            return 0.5;
        }
        return numerador/denominador;
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

            std::vector<double> c_r_succes;
            std::vector<double> f_succes;
            std::vector<double> delta_fitness;
            
            
            for(int ind=0;ind<n_inds;ind++){
                
                //MUTACION
                int rand_mem=rndm_int(rndm)%tamano_memoria;

                double cr_i=sample_cr(mem_cr[rand_mem]);
                double f_i=sample_f(mem_f[rand_mem]);
                double p_i=sample_p();
                
                n_pob[ind]=mutation_current_to_pbest_1(ind, f_i, select_pbest_index(p_i));
                
                //CROSSOVER
                n_pob[ind]->crossover(pob[ind], rndm_int(rndm)%20, cr_i);
                if(json["Clip"]==1){
                    n_pob[ind]->clip(lower_bound, upper_bound);
                }

                //EVAL
                n_pob[ind]->fitness=mlp->eval_train(n_pob[ind]);
                
                //TORNEO
                
                struct individuo *perdedor;
                if(n_pob[ind]->fitness <= pob[ind]->fitness){
                    
                    
                    if(n_pob[ind]->fitness < pob[ind]->fitness){
                        //c_r_succes.push_back(cr_i);
                        //f_succes.push_back(f_i);
                        //delta_fitness.push_back(std::abs(n_pob[ind]->fitness - pob[ind]->fitness));

                        //archivo.push_back(pob[ind]);

                        pob[ind]=n_pob[ind];
                        n_pob[ind]=nullptr;
                    }
                    else{
                        perdedor=pob[ind];
                        pob[ind]=n_pob[ind];
                        n_pob[ind]=nullptr;
                        delete perdedor;
                    }
                }
                else{
                    perdedor=n_pob[ind];
                    delete perdedor;
                    n_pob[ind] = nullptr;
                }
                while(archivo.size()>tamano_memoria){
                    int indice=rndm_int(rndm)%archivo.size();
                    delete archivo[indice];
                    archivo.erase(archivo.begin()+indice); 
                }
                
            }

            if(!c_r_succes.empty()&&!f_succes.empty()&&!delta_fitness.empty()){
                mem_cr[indice_circular%tamano_memoria]=weighted_arithmetic_mean(delta_fitness, c_r_succes);
                mem_f[indice_circular%tamano_memoria]=weighted_lehmer_mean(delta_fitness, f_succes);
                indice_circular++;
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
        //return std::make_tuple(0, std::vector<double>{}, std::vector<double>{}, std::vector<double>{}, std::vector<double>{}, std::vector<double>{}, std::vector<double>{}, std::vector<double>{}, std::vector<double>{}, std::vector<double>{}, std::vector<bool>{});
    }
};

template<typename T> thread_local std::mt19937 SHADE<T>::rndm{std::random_device{}()};

template<typename T> thread_local std::uniform_int_distribution<int> SHADE<T>::rndm_int{0, 10000000};

template<typename T> thread_local std::uniform_real_distribution<double> SHADE<T>::rndm_dbl{0.0, 1.0};

#endif