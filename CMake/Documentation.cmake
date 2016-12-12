
#
# add a target to generate API documentation with Doxygen
#
find_package(Doxygen)
SET(DOXYGEN_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/doc/Doxygen/)
if(DOXYGEN_FOUND)
	configure_file(${DOXYGEN_SOURCE_DIR}/Doxyfile.in ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)
	add_custom_target(DoxygenDoc
	${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile     
	WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
	COMMENT "Generating API documentation with Doxygen" VERBATIM
)
endif(DOXYGEN_FOUND)

#
# Build Sphinx documentation
#

FIND_PACKAGE(Sphinx)
if(NOT DEFINED SPHINX_THEME)
    set(SPHINX_THEME default)
endif()

if(NOT DEFINED SPHINX_THEME_DIR)
    set(SPHINX_THEME_DIR)
endif()

# configured documentation tools and intermediate build results
set(BINARY_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/_build")

# Sphinx cache with pickled ReST documents
set(SPHINX_CACHE_DIR "${CMAKE_CURRENT_BINARY_DIR}/_doctrees")

# Sphinx Source Dir
SET(SPHINX_SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/doc/Sphinx")

# HTML output directory
set(SPHINX_HTML_DIR "${SPHINX_SOURCE_DIR}/html/")

# Breather link
SET(BREATHE_SOURCE "/home/christian/breathe-0.7.5/")

configure_file(
    "${SPHINX_SOURCE_DIR}/conf.py.in"
    "${SPHINX_SOURCE_DIR}/conf.py"
    @ONLY)

SET(SPHINX_EXECUTABLE "/usr/local/bin/sphinx-build")

add_custom_target(SphinxDoc ALL
    ${SPHINX_EXECUTABLE}
    -q -b html
    "${SPHINX_SOURCE_DIR}"
    "${SPHINX_HTML_DIR}"
    COMMENT "Building HTML documentation with Sphinx")
