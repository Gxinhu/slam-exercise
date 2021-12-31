add_library(slam SHARED feature.cpp mappoint.cpp feature.cpp map.cpp viewer.cpp
                        frontend.cpp)
target_link_libraries(slam ${THIRD_PARTY_LIBS})
