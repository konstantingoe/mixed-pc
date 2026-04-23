# Outset

This repo now implements the PC algorithm for arbitrary mixed data. Great job!

## New to-do's

1. Let's implement the option to include prior knowledge.
   1. The first and easiest means of inserting prior knowledge is by white and black listing edges. This then simply corresponds to never touching an edge if whitelisted or removing all blacklisted edges right from the start.
   2. The second a little more complex way of including prior knowledge is by assuming some layered (or temporal) structure. In the `~/Projects/gresit/gresit/group_pc.py` file there is already an implementation of layered/temporal prior knowledge in the context of the more general group_pc. This essentially boils down to only considering edges and conditioning sets that don't violate the layering/temporal order. Feel free to change things up however if you believe there is a better way to do this.
   3. That's all the prior knoweldge I can think of currently but if you are aware of more strategies feel free to prompt me.
   4. Finish by writing tests and possibly some speed and accuracy insights when different types of prior knowledge is available.
