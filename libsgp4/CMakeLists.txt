set(SRCS
    CoordGeodetic.cc
    CoordTopocentric.cc
    DateTime.cc
    DecayedException.cc
    Eci.cc
    Globals.cc
    Observer.cc
    OrbitalElements.cc
    SGP4.cc
    SGP4Batch.cc
    JitPropagator.cc
    LinearBVH.cc
    SatelliteException.cc
    SolarPosition.cc
    TimeSpan.cc
    Tle.cc
    TleException.cc
    Util.cc
    Vector.cc)

  set(INCS
     CoordGeodetic.h
     CoordTopocentric.h
     DateTime.h
     DecayedException.h
     Eci.h
     Globals.h
     Observer.h
     OrbitalElements.h
     SatelliteException.h
     SGP4Batch.h
     JitPropagator.h
     LinearBVH.h
     SpatialHash.h
     SGP4.h
     SolarPosition.h
     TimeSpan.h
     TleException.h
     Tle.h
     Util.h
     Vector.h
     )

add_library(sgp4 STATIC ${SRCS} ${INCS})
add_library(sgp4s SHARED ${SRCS} ${INCS})
target_link_libraries(sgp4 ${CMAKE_DL_LIBS})
target_link_libraries(sgp4s ${CMAKE_DL_LIBS})
install( TARGETS sgp4s LIBRARY DESTINATION lib )
install( FILES ${INCS} DESTINATION include/libsgp4 )
