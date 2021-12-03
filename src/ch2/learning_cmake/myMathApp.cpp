#include <iostream>
#include "mymath.h"

int main(int argc, char **argv)
{
    double a = add(1.1, 2.2);
    int b = add(1, 2);
    std::cout << "1.1 + 2.2 =" << a << std::endl;
    std::cout << "1 + 2 =" << b << std::endl;
    return 0;
}