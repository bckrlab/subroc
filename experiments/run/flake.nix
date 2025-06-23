{
    description = "SubROC";

    inputs = {
        nixpkgs.url = "github:nixos/nixpkgs/release-25.05";
        flake-utils.url = "github:numtide/flake-utils";
    };

    outputs = { self, nixpkgs, flake-utils }:
        flake-utils.lib.eachDefaultSystem (system:
            let
                pkgs = import nixpkgs {
                    inherit system;
                };

                texliveCustom = pkgs.texlive.combine {
                    inherit (pkgs.texlive)
                        scheme-basic
                        cm-super
                        dvipng
                        collection-latexextra
                        collection-fontsrecommended
                        luatex;
                };
            in {
                packages.default = pkgs.buildEnv {
                    name = "runtime-env";
                    paths = [
                        pkgs.perl
                        pkgs.curl
                        pkgs.wget
                        pkgs.fontconfig
                        pkgs.cacert
                        pkgs.gnupg
                        pkgs.tmux
                        pkgs.time
                        texliveCustom
                    ];
                };
            }
        );
}