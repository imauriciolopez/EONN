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
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <cmath>
#include <algorithm>
#include <cstdlib>

#include <omp.h>

int main(int argc, char* argv[]){
    //cargar información
    std::ifstream file(argv[1]);
    if(!file){
        throw std::runtime_error("NO SE ENCONTRÓ EL JSON");
    }
    nlohmann::json json;
    file>>json;

    int n_hilos=json["N hilos"];
    int n_archivos=json["N archivos"];
    std::vector<std::string> rutas=json["Archivos"].get<std::vector<std::string> >();

    omp_set_num_threads(n_hilos);
    #pragma omp parallel for schedule(dynamic)
    for(int i=0;i<n_archivos;i++){
        std::ifstream file(rutas[i].c_str());
        nlohmann::json json_;
        file>>json_;
        
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

        if(json_["Data type"]=="double") resultados=experimento<double>(json_);
        else if(json_["Data type"]=="float") resultados=experimento<float>(json_);
        else if(json_["Data type"]=="int") resultados=experimento<int>(json_);
        else if(json_["Data type"]=="unsigned int") resultados=experimento<unsigned int>(json_);
        else if(json_["Data type"]=="bool") resultados=experimento<bool>(json_);
        else throw std::runtime_error("NO SE ENCONTRÓ EL TIPO DE DATO ESPECIFICADO EN EL JSON, terminando programa");

        guardar_resultados(resultados, json_);
    }
}

/*
    hello_world_local();
    hello_world_utils();
    hello_world_CMSA();
    hello_world_ED();
    hello_world_SHADE();
    hello_world_PSO();
    
*/

