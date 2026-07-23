#ifndef PROJECT_CONFIG_HPP
#define PROJECT_CONFIG_HPP

#include "params.hpp"
#include <string>

// Load a web_monitor JSON file. Missing fields retain the compiled defaults.
bool load_project_config(const std::string &path, prj_params &params, std::string &error);

#endif
