{

description = "(pronounced \"except\") A pseudo-golfing language based on exceptions";

inputs = {
  nixpkgs = {
    type = "github";
    owner = "NixOS";
    repo = "nixpkgs";
    ref = "release-22.11";
  };
  flake-utils = {
    type = "github";
    owner = "numtide";
    repo = "flake-utils";
  };
};

outputs = { self, nixpkgs, flake-utils }:
  flake-utils.lib.eachDefaultSystem (system:
    let
      pkgs = nixpkgs.legacyPackages.${system};
    in {
      packages = rec {
        default = with pkgs.python310Packages;
          buildPythonApplication {
            pname = "x7";
            version = "0.0.1";
            format = "pyproject";
            src = self;
            nativeBuildInputs = [ setuptools-scm ];
          };
      };
    }
  );

}
