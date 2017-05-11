{ buildPythonPackage
, pytest
, cython
, cytoolz
, multipledispatch
, numpy
, noisy
}:

buildPythonPackage rec {
  name = "streaming-${version}";
  version = "dev";

  src = ./.;

  checkInpus = [ pytest ];
  buildInputs = [ cython ];
  propagatedBuildInputs = [ cytoolz multipledispatch numpy noisy ];

  doCheck = false;
}
