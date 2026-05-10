{
  description = "Flappy Bird Agent - Python project";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.11";
  };

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };
      python = pkgs.python311;
      pythonPackages = python.pkgs;
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        name = "flappy-bird-agent";

        KERAS_BACKEND = "jax";
        venvDir = "./.venv";

        buildInputs = [
          python
          pythonPackages.venvShellHook
          pkgs.stdenv.cc.cc.lib
        ];

        postVenvCreation = ''
          unset SOURCE_DATE_EPOCH
          pip install --upgrade pip
          pip install jax jaxlib
          pip install keras
          pip install numpy optree
          pip install ipython pgmpy tqdm fastapi uvicorn nltk websockets
        '';

        postShellHook = ''
          unset PYTHONPATH
          unset SOURCE_DATE_EPOCH
        '';
      };
    };
}
