#include <iostream>

#ifndef LOCAL_HPP
#define LOCAL_HPP

#include <fstream>
#include <string>

#include <type_traits>
#include <../nlohmann/json.hpp>
#include <iostream>
#include <cmath>
#include <iterator>
#include <random>
#include <vector>
#include <unordered_map>
#include <stdexcept>
#include <ctime>

#include <typeinfo>

void hello_world_local();

//enum del tipo de problema
enum class problem_type{
    binary_classification,
    multiclass_classification,
    multilabel_classification
};

//definición de individuo
struct individuo{
    //atributos individuales
    int n_capas;
    std::vector<int> estructura;
    std::vector<std::vector<std::vector<double> > > w;
    std::vector<std::vector<double> > b;
    std::vector<double(*)(double, double)> activaciones;
    std::vector<double> bias_decision;
    double lower_bound;
    double upper_bound;
    double fitness;
    unsigned int seed;

    //atributos para Evolución Diferencial
    double F;
    double C_r;

    static thread_local std::mt19937 rndm;
    static thread_local std::uniform_int_distribution<int>     rndm_int;
    static thread_local std::uniform_real_distribution<double> rndm_dbl;

    //atributos para PSO
    individuo(int n_capas=0, 
              std::vector<int> estructura={}, 
              double lower_bound=0.0, 
              double upper_bound=0.0,
              std::vector<std::vector<std::vector<double> > > w={{{}}}, 
              std::vector<std::vector<double> > b={{}}, 
              std::vector<double(*)(double, double)> activaciones_={}, 
              std::vector<double> bias_decision={},
              double fitness=0, 
              unsigned int seed=27);

    void set_atributos_ED(double F_, double C_r_);

    individuo* operator+(const individuo* otro) const;
    
    individuo* operator-(const individuo* otro) const;
    
    individuo* operator*(const double fact) const;
    
    individuo* operator/(const double fact) const;
    
    bool operator<(const individuo& otro) const;

    bool operator<=(const individuo& otro) const;

    bool operator>(const individuo& otro) const;

    bool operator>=(const individuo& otro) const;
    
    void clip(double lower_bound, double upper_bound);

    void crossover(struct individuo *otro, int igual, double c_r=-1.0);

    void ver(bool w_=false, bool b_=false, bool bias_decision_=false);

    std::vector<std::vector<std::vector<double> > > w_random(int n_capas, std::vector<int> estructura);

    std::vector<std::vector<double> > b_random(int n_capas, std::vector<int> estructura);

    std::vector<double> b_decision_random(int n_capas, std::vector<int> estructura);

    double random_bounded();
};

//definicion del tipo de dato
template<typename T> struct dato{
    int n_input;
    std::vector<T> x;

    int n_output;
    std::vector<bool> y;
    problem_type problem;

    dato(std::vector<T> x={}, 
         std::vector<bool> y={}, 
         problem_type problem=problem_type::binary_classification):
         x(x), 
         y(y),
         problem(problem)
         {n_input=x.size();n_output=y.size();}

    void ver(){
        std::cout<<"n_input: "<<n_input<<std::endl;
        for(int i=0;i<n_input-1;i++){
            std::cout<<x[i]<<", ";
        }
        std::cout<<x[n_input-1]<<std::endl;

        std::cout<<"n_output: "<<n_output<<std::endl;
        for(int i=0;i<n_output-1;i++){
            std::cout<<y[i]<<", ";
        }
        std::cout<<y[n_output-1]<<std::endl;

        if(problem==problem_type::binary_classification){
            std::cout<<"Problem type: Binary classification"<<std::endl;
        }
        else if(problem==problem_type::multiclass_classification){
            std::cout<<"Problem type: Multiclass classification"<<std::endl;
        }
        else{
            std::cout<<"Problem type: Multilabel classification"<<std::endl;
        }
    }
};

//fución para lectura de datos
template<typename T> std::tuple<T, T, std::vector<struct dato<T>*>> leer_datos(std::string dataset, std::string tipo, int verbose=-1){
    std::string ruta="../datasets/"+dataset+"/"+tipo+".csv";

    std::ifstream archivo(ruta.c_str());
    if(!archivo){
        throw std::runtime_error("NO SE ENCONTRÓ EL ARCHIVO");
    }
    std::string linea;

    int data_size=-1;
    std::string data_type="None";
    int n_input_data=-1;
    int n_categories=-1;
    std::string problem_type="None";
    T minimo, maximo;

    size_t pos;

    int buff_int;
    double buff_double;
    bool buff_bool;

    while(data_size==-1||
          data_type=="None"||
          n_input_data==-1||
          n_categories==-1||
          problem_type=="None"){
        std::getline(archivo, linea);
        pos=linea.find("#");
        if(pos!=std::string::npos) continue;

        pos=linea.find("=");
        std::string dato=linea.substr(0, pos);
        std::string lectura=linea.substr(pos+1);

        if(dato=="data_size"){
            data_size=stoi(lectura);
        }
        else if(dato=="data_type"){
            if(!lectura.empty()&&lectura.front()=='"'){
                lectura.erase(0, 1);
            }
            if(!lectura.empty() && lectura.back()=='"'){
                lectura.pop_back();
            }
            data_type=lectura;
        }
        else if(dato=="n_input_data"){
            n_input_data=stoi(lectura);
        }
        else if(dato=="n_categories"){
            n_categories=stoi(lectura);
        }
        else if(dato=="problem_type"){
            if(!lectura.empty()&&lectura.front()=='"'){
                lectura.erase(0, 1);
            }
            if(!lectura.empty() && lectura.back()=='"'){
                lectura.pop_back();
            }
            problem_type=lectura;
        }
    }

    if(verbose!=-1){
        std::cout<<"DATOS INICIALES LEIDOS:"<<std::endl;
        std::cout<<"data_size: "<<data_size<<std::endl;
        std::cout<<"data_type: "<<data_type<<std::endl;
        std::cout<<"n_input_data: "<<n_input_data<<std::endl;
        std::cout<<"n_categories: "<<n_categories<<std::endl;
        std::cout<<"problem_type: "<<problem_type<<std::endl;
    }

    //leyendo dato por dato
    std::vector<struct dato<T>*> new_data(data_size);
    for(int data=0;data<data_size;data++){
        if(verbose!=-1&&data%verbose==0){
            std::cout<<"DATO: "<<data<<std::endl;
        }

        std::getline(archivo, linea);
        pos=linea.find("#");
        if(pos!=std::string::npos){
            data--;
            continue;
        }

        std::istringstream l(linea);
        //creando el "dato"

        if(verbose!=-1&&data%verbose==0){
            std::cout<<"ESTRUCTURA CREADA"<<std::endl;
        }

        std::vector<T> entrada(n_input_data);
        std::vector<bool> salida((problem_type=="Binary classification")?(1):(n_categories));

        //leyendo los datos de salida
        if(problem_type=="Binary classification"){
            l>>buff_int;
            salida[0]=(bool)buff_int;
        }
        else if(problem_type=="Multiclass classification"){
            l>>buff_int;
            for(int i=0;i<n_categories;i++){
                salida[i]=(buff_int==i)?(true):(false);
            }
        }
        else if(problem_type=="Multilabel classification"){
            for(int i=0;i<n_categories;i++){
                l>>buff_int;
                salida[i]=(bool)buff_int;
            }
        }
        
        if(verbose!=-1&&data%verbose==0){
            std::cout<<"SALIDA LEIDA "<<n_categories<<std::endl;
            //for(int i=0;i<()?(1):(n_categories);i++){
            if(problem_type=="Binary classification"){
                std::cout<<salida[0]<<std::endl;
            }
            else{
                for(int i=0;i<n_categories;i++){
                    std::cout<<salida[i]<<", ";
                }
                std::cout<<std::endl;
            }

        }

        //leyendo los datos de entrada
        for(int i=0;i<n_input_data;i++){
            if(data_type=="int"){
                l>>buff_int;
                entrada[i]=buff_int;
                if(minimo>buff_int){
                    minimo=buff_int;
                }
                else if(maximo<buff_int){
                    maximo=buff_int;
                }
            }
            else if(data_type=="double"||data_type=="float"){
                l>>buff_double;
                entrada[i]=buff_double;
                if(minimo>buff_double){
                    minimo=buff_double;
                }
                else if(maximo<buff_double){
                    maximo=buff_double;
                }
            }
            else if(data_type=="bool"){
                l>>buff_bool;
                entrada[i]=buff_bool;
                if(minimo>buff_bool){
                    minimo=buff_bool;
                }
                else if(maximo<buff_bool){
                    maximo=buff_bool;
                }
            }
        }

        if(verbose!=-1&&data%verbose==0){
            std::cout<<"ENTRADA LEIDA"<<std::endl;
            for(int i=0;i<n_input_data;i++){
                std::cout<<entrada[i]<<", ";
            }
            std::cout<<std::endl;
        }

        if(problem_type=="Binary classification"){
            new_data[data]=new dato<T>(entrada, salida, problem_type::binary_classification);
        }
        
        else if(problem_type=="Multiclass classification"){
            new_data[data]=new dato<T>(entrada, salida, problem_type::multiclass_classification);
        }
        
        else if(problem_type=="Multilabel classification"){
            new_data[data]=new dato<T>(entrada, salida, problem_type::multilabel_classification);
        }

        if(verbose!=-1&&data%verbose==0){
            std::cout<<"LECTURA TERMINADA"<<std::endl;
        }
    }
    archivo.close();

    return std::make_tuple(minimo, maximo, new_data);
}

//función para borrar los datos leidos
template<typename T> void borrar_datos(std::vector<struct dato<T>*> datos){
    if(datos.empty()){
        return;
    }

    for(int i=0;i<datos.size();i++){
        delete datos[i];
        datos[i]=nullptr;
    }
}

//multilayer perceptron es excelente para tareas de clasificacion, es decir, la ultima capa
//tiene la misma cantidad de neuronas y categorías, y la salida de cada neurona es binaria
template<typename T>
class MLP{
public:
    //datos solo de entrenamiento
    std::vector<struct dato<T>*> entrenamiento;
    //datos solo de validación
    std::vector<struct dato<T>*> validacion;

    //función de fitness
    using fitness_func = double (MLP<T>::*)(const std::vector<double>&, const std::vector<bool>&, int, double);
    fitness_func fit_func;

    //constructor
    MLP(fitness_func fit_func,
        std::vector<struct dato<T>*> entrenamiento, 
        std::vector<struct dato<T>*> validacion):
        entrenamiento(entrenamiento), 
        validacion(validacion), 
        fit_func(fit_func){}

    void ver(){
        std::cout<<"n datos entrenamiento: "<<entrenamiento.size()<<std::endl;
        std::cout<<"n datos validacion: "<<validacion.size()<<std::endl;
    }

    //funciones de activación
    static double tanh(double x, double offset=0.0){
        return std::tanh(x);
    }
    
    static double sigmoid(double x, double offset=0.0){
        return 1.0 / (1.0 + exp(-((x>50)?(50):((x<-50)?(50):(x)))));
    }

    static double igual(double x, double offset=0.0){
        return x;
    }

    static double ReLU(double x, double offset=0.0){
        return std::max(offset, x);
    }

    static double step_function(double x, double offset=0.0){
        return (x>offset)?(1.0):(0.0);
    }

    //softmax
    static std::vector<double> softmax(std::vector<double>& output, int n){
        std::vector<double> res(n);

        double max_val = output[0];
        for(int i = 1; i < n; i++){
            if(output[i] > max_val){
                max_val = output[i];
            }
        }

        double suma=0.0;
        for(int i=0;i<n;i++){
            res[i]=std::exp(output[i]-max_val);
            suma+=res[i];
        }
        for(int i=0;i<n;i++){
            res[i]/=suma;
        }

        return res;
    }

    //funciones de fitness
    double binary_cross_entropy(const std::vector<double>& y_pred, const std::vector<bool>& y_true, int n_datos, double eps=1e-12){
        double loss=0.0;

        std::vector<double> y_pred_(n_datos);
        for(int i=0;i<n_datos;i++){
            if(y_pred[i]<eps){
                y_pred_[i]=eps;
            }
            else if(y_pred[i]>1.0-eps){
                y_pred_[i]=1.0-eps;
            }
            else{
                y_pred_[i]=y_pred[i];
            }
            loss-=((double)y_true[i]*log(y_pred_[i]))+(1.0-(double)y_true[i])*log(1.0-y_pred_[i]);
        }
        loss/=n_datos;
        return loss;
    }

    double categorical_cross_entropy(const std::vector<double>& y_pred, const std::vector<bool>& y_true, int n_datos, double eps=1e-12){
        double loss=0.0;

        std::vector<double> y_pred_(n_datos);
        for(int i=0;i<n_datos;i++){
            if(y_pred[i]<eps){
                y_pred_[i]=eps;
            }
            else if(y_pred[i]>1.0-eps){
                y_pred_[i]=1.0-eps;
            }
            else{
                y_pred_[i]=y_pred[i];
            }
            loss-=((double)y_true[i]*log(y_pred_[i]));
        }
        
        return loss;
    }

    double accuracy(const std::vector<double>& y_pred, const std::vector<bool>& y_true, int n_datos, double eps=1e-12){
        double aciertos=0.0;

        for(int i=0;i<n_datos;i++){
            if((y_pred[i]>0.5)==y_true[i]){
                aciertos-=1.0;
            }
        }
        
        return aciertos/(1.0*n_datos);
    }

    //evaluadores
    std::vector<double> forward(struct individuo *ind, struct dato<T> *dato){  
        //primero vemos si la estructura coincide con el dato
        if(ind->estructura[0]!=dato->n_input||ind->estructura[ind->n_capas-1]!=dato->n_output){
            throw std::runtime_error("NO COINCIDE LA ESTRUCTURA CON LOS DATOS");
        }
        
        //creamos la matriz donde se van a guardar todas las salidas de cada neurona
        std::vector<std::vector<double> > salidas(ind->n_capas);
        for(int i=0;i<ind->n_capas;i++){
            salidas[i]=std::vector<double>(ind->estructura[i], 0.0);
        }
        
        //primera capa
        //iteramos sobre todas las neuronas de la primera capa
        //cada una solo tendrá una entrada
        for(int neurona=0;neurona<ind->estructura[0];neurona++){
            salidas[0][neurona]=ind->activaciones[0]((double)(dato->x[neurona])*ind->w[0][neurona][0]+ind->b[0][neurona], 0.0);
        }
        
        //capas de enmedio
        //iteramos sobre todas las capas (dado por la cantidad de capas del individuo)
        for(int capa=1;capa<ind->n_capas;capa++){
            //iteramos sobre cada neurona (dado por la cantidad de neuronas de cada capa)
            for(int neurona=0;neurona<ind->estructura[capa];neurona++){
                //iteramos sobre cada peso (dado por la cantidad de neuronas de la capa anterior)
                for(int peso=0;peso<ind->estructura[capa-1];peso++){
                    //a cada salida de la neurona tenemos que hacer sumatoria
                    salidas[capa][neurona]+=salidas[capa-1][peso]*ind->w[capa][neurona][peso];
                }
                //calculamos la función de activación
                salidas[capa][neurona]=ind->activaciones[capa](salidas[capa][neurona]+ind->b[capa][neurona], (capa!=ind->n_capas-1)?(0.0):(ind->bias_decision[neurona]));
            }
        }

        std::vector<double> salida_final(dato->n_output);
        for(int i=0;i<dato->n_output;i++){
            salida_final[i]=salidas[ind->n_capas-1][i];
        }

        /*
        for(int i=0;i<dato->n_output;i++){
            std::cout<<"-- "<<salida_final[i]<<" --"<<std::endl;
        }
        std::cout<<std::endl;*/
        
        //si el problema es multiclass, entonces necesitamos softmax
        if(dato->problem==problem_type::multiclass_classification){
            salida_final=softmax(salida_final, dato->n_output);
        }

        

        return salida_final;
    }

    double eval_train(struct individuo *ind){
        double sum=0.0;
        for(int i=0;i<entrenamiento.size();i++){
            std::vector<double> res=forward(ind, entrenamiento[i]);
            sum+=(this->*fit_func)(res, entrenamiento[i]->y, entrenamiento[i]->n_output, 1e-12);
        }
        return sum;
    }

    double eval_test(struct individuo *ind){
        double sum=0.0;
        for(int i=0;i<validacion.size();i++){
            std::vector<double> res=forward(ind, validacion[i]);
            sum+=(this->*fit_func)(res, entrenamiento[i]->y, entrenamiento[i]->n_output, 1e-12);
        }
        return sum;
    }

    double eval_train_accuracy(struct individuo *ind){
        double sum=0.0;
        for(int i=0;i<entrenamiento.size();i++){
            std::vector<double> res=forward(ind, entrenamiento[i]);
            sum-=accuracy(res, entrenamiento[i]->y, entrenamiento[i]->n_output, 1e-12);
        }
        return sum/(entrenamiento.size()*1.0);
    }

    double eval_test_accuracy(struct individuo *ind){
        double sum=0.0;
        for(int i=0;i<validacion.size();i++){
            std::vector<double> res=forward(ind, validacion[i]);
            sum-=accuracy(res, entrenamiento[i]->y, entrenamiento[i]->n_output, 1e-12);
        }
        return sum/(validacion.size()*1.0);
    }
};

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
                               succes);
    }
}; 

template<typename T> thread_local std::mt19937 ED<T>::rndm{std::random_device{}()};

template<typename T> thread_local std::uniform_int_distribution<int> ED<T>::rndm_int{0, 10000000};

template<typename T> thread_local std::uniform_real_distribution<double> ED<T>::rndm_dbl{0.0, 1.0};

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
                               succes);
        //return std::make_tuple(0, std::vector<double>{}, std::vector<double>{}, std::vector<double>{}, std::vector<double>{}, std::vector<double>{}, std::vector<double>{}, std::vector<double>{}, std::vector<double>{}, std::vector<double>{}, std::vector<bool>{});
    }
};

template<typename T> thread_local std::mt19937 SHADE<T>::rndm{std::random_device{}()};

template<typename T> thread_local std::uniform_int_distribution<int> SHADE<T>::rndm_int{0, 10000000};

template<typename T> thread_local std::uniform_real_distribution<double> SHADE<T>::rndm_dbl{0.0, 1.0};

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
                               succes);
        
    }
}; 

template<typename T> thread_local std::mt19937 PSO<T>::rndm{std::random_device{}()};

template<typename T> thread_local std::uniform_int_distribution<int> PSO<T>::rndm_int{0, 10000000};

template<typename T> thread_local std::uniform_real_distribution<double> PSO<T>::rndm_dbl{0.0, 1.0};

template<typename T> std::tuple<unsigned int,
                                std::vector<double>, 
                                std::vector<double>, 
                                std::vector<double>,
                                std::vector<double>, 
                                std::vector<double>, 
                                std::vector<double>, 
                                std::vector<double>, 
                                std::vector<double>,
                                std::vector<double>, 
                                std::vector<bool> > experimento(nlohmann::json json){
    
    //leyendo los datos
    if(json["Verbose"]>0) std::cout<<"LEYENDO DATOS"<<std::endl;
    std::tuple<T, T, std::vector<struct dato<T>*> > datos_comp_train=leer_datos<T>(json["Dataset"], "train");
    std::tuple<T, T, std::vector<struct dato<T>*> > datos_comp_test=leer_datos<T>(json["Dataset"], "test");
    std::vector<struct dato<T>*> datos_train=get<2>(datos_comp_train);
    std::vector<struct dato<T>*> datos_test=get<2>(datos_comp_test);
    if(json["Verbose"]>0) std::cout<<"DATOS LEÍDOS"<<std::endl;
    class MLP<T> *mlp;

    if(json["Normalizacion"]==1){
        if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>){
            T min_train=get<0>(datos_comp_train), max_train=get<1>(datos_comp_train);
            for(int i=0;i<datos_train.size();i++){
                for(int j=0;j<datos_train[i]->x.size();j++){
                    datos_train[i]->x[j]-=min_train;
                    datos_train[i]->x[j]/=(max_train-min_train);
                }
            }

            T min_test=get<0>(datos_comp_test), max_test=get<1>(datos_comp_test);
            for(int i=0;i<datos_test.size();i++){
                for(int j=0;j<datos_test[i]->x.size();j++){
                    datos_test[i]->x[j]-=min_test;
                    datos_test[i]->x[j]/=(max_test-min_test);
                }
            }
        }
    }
    
    //creando la estructura del MLP
    //opciones:
    //&MLP<T>::categorical_cross_entropy -> para clasificación multiclase
    //&MLP<T>::binary_cross_entropy      -> para clasificación binaria y multilabel
    //&MLP<T>::accuracy                  -> para cualquier tipo

    if(json["Verbose"]>0) std::cout<<"CREANDO LA ESTRUCTURA MLP"<<std::endl;
    if(json["Fitness"]=="accuracy"){
        mlp=new MLP<T>(&MLP<T>::accuracy, //función de fitness
                            datos_train,            //todos los datos
                            datos_test);
    }
    else if(json["Fitness"]=="categorical"){
        mlp=new MLP<T>(&MLP<T>::categorical_cross_entropy, //función de fitness
                            datos_train,                             //todos los datos
                            datos_test);
    }
    else if(json["Fitness"]=="binary"){
        mlp=new MLP<T>(&MLP<T>::binary_cross_entropy, //función de fitness
                            datos_train,                        //todos los datos
                            datos_test);
    }
    else{
        borrar_datos(datos_train);
        borrar_datos(datos_test);
        throw std::runtime_error("NO SE ENCONTRÓ EL TIPO DE FITNESS, terminando programa");
    }
    if(json["Verbose"]>0) std::cout<<"ESTRUCTURA CREADA\nCREANDO OPTIMIZADOR"<<std::endl;
    
    std::vector<std::string> activaciones_str=json["Activaciones"].get<std::vector<std::string> >();
    std::vector<double(*)(double, double)> activaciones(json["N capas"]);
    for(int i=0;i<json["N capas"];i++){
        if(activaciones_str[i]=="tanh"){
            activaciones[i]=MLP<T>::tanh;
        }
        else if(activaciones_str[i]=="sigmoid"){
            activaciones[i]=MLP<T>::sigmoid;
        }
        else if(activaciones_str[i]=="step"){
            activaciones[i]=MLP<T>::step_function;
        }
        else if(activaciones_str[i]=="nada"){
            activaciones[i]=MLP<T>::igual;
        }
        else if(activaciones_str[i]=="ReLU"){
            activaciones[i]=MLP<T>::ReLU;
        }
        else{
            borrar_datos(datos_train);
            borrar_datos(datos_test);
            throw std::runtime_error("NO SE ENCONTRÓ EL TIPO DE ACTIVACION, terminando programa");
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
               std::vector<bool> > resultados;
    
    //inicializando optimizador
    if(json["Optimizador"]=="DE"){
        ED <T>ed(mlp,             //MLP object
                 json["Seed"],    //seed
                 json["Lower bound"], 
                 json["Upper bound"], 
                 json["N inds"],  //n_inds
                 json["N iters"], //n_iters
                 json["F"],       //F
                 json["C_r"]);    //C_r

        if(json["Verbose"]>0) std::cout<<"OPTIMIZADOR CREADO\nINICIALIZANDO POBLACIÓN"<<std::endl;
        
        ed.inicializar_poblacion(json["N capas"],               //n_capas
                                json["Estructura"].get<std::vector<int> >(), //estructura
                                activaciones); //activaciones por capa
        
        if(json["Verbose"]>0) std::cout<<"POBLACIÓN INICIALIZADA\nOPTIMIZANDO"<<std::endl;
        
        resultados=ed.optimize(json["Verbose"], json);
        
        if(json["Verbose"]>0) std::cout<<"OPTIMIZACIÓN FINALIZADA"<<std::endl;
        ed.liberar_poblacion();
    }
    
    else if(json["Optimizador"]=="SHADE"){
        SHADE <T>shade(mlp,            //MLP object
                 json["Seed"],   //seed
                 json["Lower bound"], 
                 json["Upper bound"], 
                 json["N inds"], //n_inds
                 json["N iters"],//n_iters
                 json["F"],      //F
                 json["C_r"]);   //C_r

        if(json["Verbose"]>0) std::cout<<"OPTIMIZADOR CREADO\nINICIALIZANDO POBLACIÓN"<<std::endl;
        
        shade.inicializar_poblacion(json["N capas"],               //n_capas
                                json["Estructura"].get<std::vector<int> >(), //estructura
                                activaciones); //activaciones por capa
        
        if(json["Verbose"]>0) std::cout<<"POBLACIÓN INICIALIZADA\nOPTIMIZANDO"<<std::endl;
        
        resultados=shade.optimize(json["Verbose"], json);
        
        if(json["Verbose"]>0) std::cout<<"OPTIMIZACIÓN FINALIZADA"<<std::endl;
        shade.liberar_poblacion();
    }
    
    else if(json["Optimizador"]=="PSO"){
        PSO <T>pso(mlp,            //MLP object
                 json["Seed"],   //seed
                 json["Lower bound"], 
                 json["Upper bound"], 
                 json["N inds"], //n_inds
                 json["N iters"],//n_iters
                 json["F"],      //F
                 json["C_r"]);   //C_r

        if(json["Verbose"]>0) std::cout<<"OPTIMIZADOR CREADO\nINICIALIZANDO POBLACIÓN"<<std::endl;
        
        pso.inicializar_poblacion(json["N capas"],               //n_capas
                                json["Estructura"].get<std::vector<int> >(), //estructura
                                activaciones); //activaciones por capa
        
        if(json["Verbose"]>0) std::cout<<"POBLACIÓN INICIALIZADA\nOPTIMIZANDO"<<std::endl;
        
        resultados=pso.optimize(json["Verbose"], json);
        
        if(json["Verbose"]>0) std::cout<<"OPTIMIZACIÓN FINALIZADA"<<std::endl;

        pso.liberar_poblacion();
    }
    
    else{
        borrar_datos(datos_train);
        borrar_datos(datos_test);
        throw std::runtime_error("NO SE ENCONTRÓ EL TIPO DE OPTIMIZADOR, terminando programa");
    }
    
    borrar_datos(datos_train);
    borrar_datos(datos_test);
    delete mlp;

    return resultados;
    //return std::make_tuple(0, std::vector<double>{}, std::vector<double>{}, std::vector<double>{}, std::vector<double>{}, std::vector<double>{}, std::vector<double>{}, std::vector<double>{}, std::vector<double>{}, std::vector<double>{}, std::vector<bool>{});
    
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
                                std::vector<bool> >resultados, 
                                nlohmann::json json);

#endif