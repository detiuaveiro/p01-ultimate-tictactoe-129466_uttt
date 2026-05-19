{
  description = "Ultimate Tic-Tac-Toe AI Agent";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.11";
  };

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        config.allowUnfree = true;
      };
      python = pkgs.python311;
      pythonPackages = python.pkgs;
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        name = "uttt-ai-agent";

        venvDir = "./.venv";

        buildInputs = [
          python
          pythonPackages.venvShellHook
          pkgs.stdenv.cc.cc.lib
          pkgs.zlib
          pkgs.docker
          pkgs.docker-compose
        ];

        postVenvCreation = ''
          unset SOURCE_DATE_EPOCH
          pip install --upgrade pip
          pip install torch numpy tqdm websockets pytest
        '';

        postShellHook = ''
          unset PYTHONPATH
          unset SOURCE_DATE_EPOCH

          export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath [ pkgs.stdenv.cc.cc.lib pkgs.zlib ]}:$LD_LIBRARY_PATH"

          if [ -d "/run/opengl-driver/lib" ]; then
            export LD_LIBRARY_PATH="/run/opengl-driver/lib:$LD_LIBRARY_PATH"
          fi
        '';
      };
    };
}
