#ifndef _0be72e07_d229_4b79_981d_61812416f783
#define _0be72e07_d229_4b79_981d_61812416f783

#include <string>
#include "stringtools/dump.hpp"

namespace str{

class join
{
    std::string const seperator;

public:
    join(std::string const& seperator):
        seperator(seperator)
    {}

    template<typename... Args>
    std::string operator () (Args const&... args) const {
        return dump(this->seperator, args...);
    }
};

}

#endif
