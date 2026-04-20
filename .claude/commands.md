This repo should implement an efficient implementation of the PC algorithm so that I can add the nonparanormal polychoric and polyserial correlations in order to handle mixed data.

Things that still need to be done:

1. Go through the project, understand what is there and check for bugs, errors, or inefficient implementations.
2. Check whether the skeleton search phase works correctly by writing exhaustive tests.
3. Check whether the latest proposals for the orientation phase are implemented. At least we need the conservative PC, the majority-rule PC and the Max-pvalue PC.
4. Write exhaustive tests to check whether each of the orientation strategies perform as expected.
5. Test against the causal-learn PC implementation and benchmark speed and performance on synthetic data.
6. When all this is done we need bring over the nonparanormal measures in particular kendalls tau or spearmans rho with the known mapping, the polychoric and polyserial correlations implemented in this repo: https://github.com/konstantingoe/mixed-gm
7. Lastly check again whether this works as expected and perform a little simulation.