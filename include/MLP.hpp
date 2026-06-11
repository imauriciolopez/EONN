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

template <typename T> class ED;
template <typename T> class SHADE;
template <typename T> class PSO;
template <typename T> class CMSA;

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
                                std::vector<double>, 
                                std::vector<bool> > experimento(nlohmann::json json){

    //CONFIGURANDO EL OBJETO DE MLP
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
    
    //CREANDO EL OBJETO DEL OPTIMIZADOR
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
    else if(json["Optimizador"]=="CMSA"){
        CMSA <T>cmsa(mlp,            //MLP object
                 json["Seed"],       //seed
                 json["Lower bound"], 
                 json["Upper bound"], 
                 json["N inds"],     //n_inds
                 json["N iters"],    //n_iters
                 json["F"],          //F
                 json["C_r"]);       //C_r

        if(json["Verbose"]>0) std::cout<<"OPTIMIZADOR CREADO\nINICIALIZANDO POBLACIÓN"<<std::endl;
        
        cmsa.inicializar_poblacion(json["N capas"],               //n_capas
                                json["Estructura"].get<std::vector<int> >(), //estructura
                                activaciones); //activaciones por capa
        
        if(json["Verbose"]>0) std::cout<<"POBLACIÓN INICIALIZADA\nOPTIMIZANDO"<<std::endl;
        
        resultados=cmsa.optimize(json["Verbose"], json);
        
        if(json["Verbose"]>0) std::cout<<"OPTIMIZACIÓN FINALIZADA"<<std::endl;

        cmsa.liberar_poblacion();
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
    //return std::make_tuple(0, std::vector<double>{}, std::vector<double>{}, std::vector<double>{}, std::vector<double>{}, std::vector<double>{}, std::vector<double>{}, std::vector<double>{}, std::vector<double>{}, std::vector<double>{}, std::vector<double>{}, std::vector<bool>{});
    
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
                                nlohmann::json json);


#endif