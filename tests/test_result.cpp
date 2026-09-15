#include "benchkit/environment.hpp"

#include <cassert>

int main()
{
    const auto env = benchkit::collect_environment();

    assert(!env.hostname.empty());
    assert(!env.kernel.empty());
    assert(!env.compiler.empty());
    assert(!env.cmake.empty());

    return 0;
}