#include "../include/utils.hpp"
#include "../include/MLP.hpp"
#include "../include/ED.hpp"
#include "../include/PSO.hpp"
#include "../include/SHADE.hpp"
#include "../include/CMSA.hpp"

#include <../nlohmann/json.hpp>
#include <iostream>
#include <iomanip>
#include <random>
#include <ctime>
#include <cstring>
#include <vector>
#include <unordered_map>
#include <cmath>
#include <algorithm>
#include <cstdlib>
#include <omp.h>

void hello_world_local(){
    std::cout<<"Hello, world! (from local)"<<std::endl;
}

individuo::individuo(int n_capas,
                     std::vector<int> estructura,
                     double lower_bound,
                     double upper_bound,
                     std::vector<std::vector<std::vector<double>>> w,
                     std::vector<std::vector<double>> b,
                     std::vector<double(*)(double, double)> activaciones_,
                     std::vector<double> bias_decision,
                     double fitness,
                     unsigned int seed): 
      upper_bound(upper_bound), 
      lower_bound(lower_bound),
      seed(seed),
      n_capas(n_capas),
      estructura(estructura),
      w(w),
      b(b),
      activaciones(activaciones_),
      bias_decision(bias_decision),
      fitness(fitness){
    F=0.0;
    C_r=0.0;
    rndm.seed(seed);
}

thread_local std::mt19937 individuo::rndm{std::random_device{}()};

thread_local std::uniform_int_distribution<int> individuo::rndm_int{0, 10000000};

thread_local std::uniform_real_distribution<double> individuo::rndm_dbl{0.0, 1.0};

double individuo::random_bounded(){
    std::uniform_real_distribution<double> dist(lower_bound, upper_bound);
    return dist(rndm);
}

void individuo::set_atributos_ED(double F_, double C_r_){
    F=F_;
    C_r=C_r_;
}

individuo* individuo::operator+(const individuo* otro) const{                                        
    std::vector<std::vector<std::vector<double> > > w_res(w.size());
    for(int i=0;i<w.size();i++){
        w_res[i]=std::vector<std::vector<double> >(w[i].size());
        for(int j=0;j<w[i].size();j++){
            w_res[i][j]=std::vector<double>(w[i][j].size());
            for(int k=0;k<w[i][j].size();k++){
                w_res[i][j][k]=w[i][j][k]+otro->w[i][j][k];
            }
        }
    }

    std::vector<std::vector<double> > b_res(b.size());
    for(int i=0;i<b.size();i++){
        b_res[i]=std::vector<double>(b[i].size());
        for(int j=0;j<b[i].size();j++){
            b_res[i][j]=b[i][j]+otro->b[i][j];
        }
    }
    
    std::vector<double> b_dec_res(bias_decision.size());
    for(int i=0;i<bias_decision.size();i++){
        b_dec_res[i]=bias_decision[i]+otro->bias_decision[i];
    }

    individuo *resultado=new individuo(n_capas, 
                                    estructura, 
                                    lower_bound, 
                                    upper_bound, 
                                    w_res, 
                                    b_res, 
                                    activaciones, 
                                    b_dec_res,
                                    0.0, 
                                    seed+(unsigned int)rndm_int(rndm));

    return resultado;
}

individuo* individuo::operator-(const individuo* otro) const{
    
    std::vector<std::vector<std::vector<double> > > w_res(w.size());
    for(int i=0;i<w.size();i++){
        w_res[i]=std::vector<std::vector<double> >(w[i].size());
        for(int j=0;j<w[i].size();j++){
            w_res[i][j]=std::vector<double>(w[i][j].size());
            for(int k=0;k<w[i][j].size();k++){
                w_res[i][j][k]=w[i][j][k]-otro->w[i][j][k];
            }
        }
    }
    

    std::vector<std::vector<double> > b_res(b.size());
    for(int i=0;i<b.size();i++){
        b_res[i]=std::vector<double>(b[i].size());
        for(int j=0;j<b[i].size();j++){
            b_res[i][j]=b[i][j]-otro->b[i][j];
        }
    }
    

    std::vector<double> b_dec_res(bias_decision.size());
    for(int i=0;i<bias_decision.size();i++){
        b_dec_res[i]=bias_decision[i]-otro->bias_decision[i];
    }
    
    individuo *resultado=new individuo(n_capas, 
                                        estructura, 
                                        lower_bound, 
                                        upper_bound, 
                                        w_res, 
                                        b_res, 
                                        activaciones, 
                                        b_dec_res,
                                        0.0);

    return resultado;
}

individuo* individuo::operator*(const double fact) const{
    std::vector<std::vector<std::vector<double> > > w_res(w.size());
    for(int i=0;i<w.size();i++){
        w_res[i]=std::vector<std::vector<double> >(w[i].size());
        for(int j=0;j<w[i].size();j++){
            w_res[i][j]=std::vector<double>(w[i][j].size());
            for(int k=0;k<w[i][j].size();k++){
                w_res[i][j][k]=w[i][j][k]*fact;
            }
        }
    }


    std::vector<std::vector<double> > b_res(b.size());
    for(int i=0;i<b.size();i++){
        b_res[i]=std::vector<double>(b[i].size());
        for(int j=0;j<b[i].size();j++){
            b_res[i][j]=b[i][j]*fact;
        }
    }
    
    std::vector<double> b_dec_res(bias_decision.size());
    for(int i=0;i<bias_decision.size();i++){
        b_dec_res[i]=bias_decision[i]*fact;
    }
    
    individuo *resultado=new individuo(n_capas, 
                                        estructura, 
                                        lower_bound, 
                                        upper_bound, 
                                        w_res, 
                                        b_res, 
                                        activaciones, 
                                        b_dec_res,
                                        0.0);

    return resultado;
}

individuo* individuo::operator/(const double fact) const{
                                        
    std::vector<std::vector<std::vector<double> > > w_res(w.size());
    for(int i=0;i<w.size();i++){
        w_res[i]=std::vector<std::vector<double> >(w[i].size());
        for(int j=0;j<w[i].size();j++){
            w_res[i][j]=std::vector<double>(w[i][j].size());
            for(int k=0;k<w[i][j].size();k++){
                w_res[i][j][k]=w[i][j][k]/fact;
            }
        }
    }

    std::vector<std::vector<double> > b_res(b.size());
    for(int i=0;i<b.size();i++){
        b_res[i]=std::vector<double>(b[i].size());
        for(int j=0;j<b[i].size();j++){
            b_res[i][j]=b[i][j]/fact;
        }
    }
    
    std::vector<double> b_dec_res(bias_decision.size());
    for(int i=0;i<bias_decision.size();i++){
        b_dec_res[i]=bias_decision[i]/fact;
    }
    
    individuo *resultado=new individuo(n_capas, 
                                        estructura, 
                                        lower_bound, 
                                        upper_bound, 
                                        w_res, 
                                        b_res, 
                                        activaciones, 
                                        b_dec_res,
                                        0.0);

    return resultado;
}

bool individuo::operator<(const individuo& otro) const{
    return fitness>otro.fitness;
}
bool individuo::operator<=(const individuo& otro) const{
    return fitness>=otro.fitness;
}
bool individuo::operator>(const individuo& otro) const{
    return fitness<otro.fitness;
}
bool individuo::operator>=(const individuo& otro) const{
    return fitness<=otro.fitness;
}

void individuo::clip(double lower_bound, double upper_bound){
    for(int i=0;i<w.size();i++){
        for(int j=0;j<w[i].size();j++){
            for(int k=0;k<w[i][j].size();k++){
                while(w[i][j][k]<lower_bound||w[i][j][k]>upper_bound){
                    if(w[i][j][k]<lower_bound){
                        w[i][j][k]=lower_bound+(lower_bound-w[i][j][k]);
                    }
                    else{
                        w[i][j][k]=upper_bound-(w[i][j][k]-upper_bound);
                    }
                }
            }
        }
    }
    for(int i=0;i<b.size();i++){
        for(int j=0;j<b[i].size();j++){
            while(b[i][j]<lower_bound||b[i][j]>upper_bound){
                if(b[i][j]<lower_bound){
                    b[i][j]=lower_bound+(lower_bound-b[i][j]);
                }
                else{
                    b[i][j]=upper_bound-(b[i][j]-upper_bound);
                }
            }
        }
    }
    for(int i=0;i<bias_decision.size();i++){
        while(bias_decision[i]<lower_bound||bias_decision[i]>upper_bound){
            if(bias_decision[i]<lower_bound){
                bias_decision[i]=lower_bound+(lower_bound-bias_decision[i]);
            }
            else{
                bias_decision[i]=upper_bound-(bias_decision[i]-upper_bound);
            }
        }
    }
}

void individuo::crossover(struct individuo *otro, int igual, double c_r){
    if(c_r<=0.0){
        c_r=C_r;
    }
    int count=0;
    for(int i=0;i<w.size();i++){
        for(int j=0;j<w[i].size();j++){
            for(int k=0;k<w[i][j].size();k++){
                if(rndm_dbl(rndm)<c_r||count++==igual){
                    w[i][j][k]=otro->w[i][j][k];
                }
            }
        }
    }
    for(int i=0;i<b.size();i++){
        for(int j=0;j<b[i].size();j++){
            if(rndm_dbl(rndm)<c_r||count++==igual){
                b[i][j]=otro->b[i][j];
            }
        }
    }
    for(int i=0;i<bias_decision.size();i++){
        if(rndm_dbl(rndm)<c_r||count++==igual){
            bias_decision[i]=otro->bias_decision[i];
        }
        count++;
    }
}

void individuo::ver(bool w_, bool b_, bool bias_decision_){
    //atributos individuales
    std::cout<<"n_capas: "<<n_capas<<std::endl;
    std::cout<<"estructura: ";
    for(int i=0;i<n_capas;i++){
        std::cout<<estructura[i]<<", ";
    }
    std::cout<<std::endl;
    if(w_){
        std::cout<<"w: "<<std::endl;
        for(int i=0;i<w.size();i++){
            for(int j=0;j<w[i].size();j++){
                for(int k=0;k<w[i][j].size();k++){
                    std::cout<<w[i][j][k]<<", ";
                }
                std::cout<<std::endl;
            }
            std::cout<<std::endl;
        }
    }
    if(b_){
        std::cout<<"b: "<<std::endl;
        for(int i=0;i<b.size();i++){
            for(int j=0;j<b[i].size();j++){
                std::cout<<b[i][j]<<", ";
            }
            std::cout<<std::endl;
        }
    }
    if(bias_decision_){
        std::cout<<"bias_decision: "<<std::endl;
        for(int i=0;i<bias_decision.size();i++){
            std::cout<<bias_decision[i]<<", ";
        }
        std::cout<<std::endl;
    }
    std::cout<<"lower_bound: "<<lower_bound<<std::endl;
    std::cout<<"upper_bound: "<<upper_bound<<std::endl;
    std::cout<<"fitness: "<<fitness<<std::endl;
    std::cout<<"F: "<<F<<std::endl;
    std::cout<<"C_r: "<<C_r<<std::endl;
}

std::vector<std::vector<std::vector<double> > > individuo::w_random(int n_capas, std::vector<int> estructura){
    std::vector<std::vector<std::vector<double> > > w(n_capas);
    
    //iteramos sobre todas las capas (dado por la cantidad de capas del individuo)
    for(int capa=0;capa<n_capas;capa++){
        w[capa]=std::vector<std::vector<double> >(estructura[capa]);

        //iteramos sobre cada neurona (dado por la cantidad de neuronas de cada capa)
        for(int neurona=0;neurona<estructura[capa];neurona++){
            w[capa][neurona]=std::vector<double>((capa==0)?(1):(estructura[capa-1]));        

            //iteramos sobre cada peso (dado por la cantidad de neuronas de la capa anterior)
            for(int peso=0;peso<((capa==0)?(1):(estructura[capa-1]));peso++){
                //asignamos de manera aleatoria entre los bounds el valor del peso
                w[capa][neurona][peso]=random_bounded();
            }
        }
    }

    return w;
}

std::vector<std::vector<double> > individuo::b_random(int n_capas, std::vector<int> estructura){
    std::vector<std::vector<double> > b(n_capas);
    for(int capa=0;capa<n_capas;capa++){
        b[capa]=std::vector<double>(estructura[capa]);
        for(int neurona=0;neurona<estructura[capa];neurona++){
            b[capa][neurona]=random_bounded();
        }
    }
    return b;
}

std::vector<double> individuo::b_decision_random(int n_capas, std::vector<int> estructura){
    std::vector<double> b_decision(estructura[n_capas-1]);
    for(int neurona=0;neurona<estructura[n_capas-1];neurona++){
        b_decision[neurona]=random_bounded();
    }
    return b_decision;
}

std::vector<double> individuo::aplanar_parametros(){
    std::vector<double> vector_plano;
    
    for(int i=0;i<w.size();i++){
        for(int j=0;j<w[i].size();j++){
            for(int k=0;k<w[i][j].size();k++){
                vector_plano.push_back(w[i][j][k]);
            }
        }
    }
    
    for(int i=0;i<b.size();i++){
        for(int j=0;j<b[i].size();j++){
            vector_plano.push_back(b[i][j]);
        }
    }

    for(int i=0;i<bias_decision.size();i++){
        vector_plano.push_back(bias_decision[i]);
    }
    
    return vector_plano;
}

void individuo::desaplanar_parametros(std::vector<double>& vector_plano) {
    int idx=0;

    for(int i=0;i<w.size();i++){
        for(int j=0;j<w[i].size();j++){
            for(int k=0;k<w[i][j].size();k++){
                w[i][j][k]=vector_plano[idx++];
            }
        }
    }

    for(int i=0;i<b.size();i++){
        for(int j=0;j<b[i].size();j++){
            b[i][j]=vector_plano[idx++];
        }
    }

    for(int i=0;i<bias_decision.size();i++){
        bias_decision[i]=vector_plano[idx++];
    }
}

void guardar_resultados(std::tuple<unsigned int,
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
                                std::vector<bool> >resultados, 
                                nlohmann::json json){

    std::ofstream archivo(json["Archivo salida"].get<std::string>().c_str()); // crea o sobrescribe el archivo

    if(!archivo.is_open()){
        throw std::runtime_error("NO SE ENCONTRÓ EL ARCHIVO DE SALIDA, los resultados no se guardaron, terminando programa");
    }

    unsigned int seed=                  get<0>(resultados);
    std::vector<double> best_loss=      get<1>(resultados);
    std::vector<double> worst_loss=     get<2>(resultados);
    std::vector<double> best_accuracy=  get<3>(resultados);
    std::vector<double> worst_accuracy= get<4>(resultados);
    std::vector<double> iteracion_mejor=get<5>(resultados);
    std::vector<double> iteracion_peor= get<6>(resultados);
    std::vector<double> promedio=       get<7>(resultados);
    std::vector<double> desvest=        get<8>(resultados);
    std::vector<double> tiempo_por_iteracion=get<9>(resultados);
    std::vector<double> poblacion_final=get<10>(resultados);
    std::vector<bool> succes=           get<11>(resultados);

    archivo<<"{\n";
    archivo<<"  \"Seed\": "<<seed<<",\n";
    archivo<<"  \"Best loss\": [";
    if(!best_loss.empty()){
        for(int i=0;i<best_loss.size()-1;i++){
            archivo<<best_loss[i]<<", ";
        }
        archivo<<best_loss[best_loss.size()-1];
    }
    
    archivo<<"],\n  \"Worst loss\": [";
    if(!worst_loss.empty()){
        for(int i=0;i<worst_loss.size()-1;i++){
            archivo<<worst_loss[i]<<", ";
        }
        archivo<<worst_loss[worst_loss.size()-1];
    }

    archivo<<"],\n  \"Best accuracy\": [";
    if(!best_accuracy.empty()){
        for(int i=0;i<best_accuracy.size()-1;i++){
            archivo<<best_accuracy[i]<<", ";
        }
        archivo<<best_accuracy[best_accuracy.size()-1];
    }
    
    archivo<<"],\n  \"Worst accuracy\": [";
    if(!worst_accuracy.empty()){
        for(int i=0;i<worst_accuracy.size()-1;i++){
            archivo<<worst_accuracy[i]<<", ";
        }
        archivo<<worst_accuracy[worst_accuracy.size()-1];
    }
    
    archivo<<"],\n  \"Best iteration\": [";
    if(!iteracion_mejor.empty()){
        for(int i=0;i<iteracion_mejor.size()-1;i++){
            archivo<<iteracion_mejor[i]<<", ";
        }
        archivo<<iteracion_mejor[iteracion_mejor.size()-1];
    }
    
    archivo<<"],\n  \"Worst iteration\": [";
    if(!iteracion_peor.empty()){
        for(int i=0;i<iteracion_peor.size()-1;i++){
            archivo<<iteracion_peor[i]<<", ";
        }
        archivo<<iteracion_peor[iteracion_peor.size()-1];
    }

    archivo<<"],\n  \"Promedio\": [";
    if(!promedio.empty()){
        for(int i=0;i<promedio.size()-1;i++){
            archivo<<promedio[i]<<", ";
        }
        archivo<<promedio[promedio.size()-1];
    }

    archivo<<"],\n  \"Desvest\": [";
    if(!desvest.empty()){
        for(int i=0;i<desvest.size()-1;i++){
            archivo<<desvest[i]<<", ";
        }
        archivo<<desvest[desvest.size()-1];
    }
    
    archivo<<"],\n  \"Tiempo por iteración\": [";
    if(!tiempo_por_iteracion.empty()){
        for(int i=0;i<tiempo_por_iteracion.size()-1;i++){
            archivo<<tiempo_por_iteracion[i]<<", ";
        }
        archivo<<tiempo_por_iteracion[tiempo_por_iteracion.size()-1];
    }

    archivo<<"],\n  \"Población final\": [";
    if(!poblacion_final.empty()){
        for(int i=0;i<poblacion_final.size()-1;i++){
            archivo<<poblacion_final[i]<<", ";
        }
        archivo<<poblacion_final[poblacion_final.size()-1];
    }
    
    archivo<<"],\n  \"Succes\": [";
    if(!succes.empty()){
        for(int i=0;i<succes.size()-1;i++){
            archivo<<succes[i]<<", ";
        }
        archivo<<succes[succes.size()-1];
    }
    
    archivo<<"]\n}";
}


