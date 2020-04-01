## Alternate Codebase

This folder contains an alternate implementation of dynamic kernel matching for classify sequences. The codebase uses only Keras to setup and run the model, making it dramatically simpler than the original code. To use this code, simply move the python scripts into the parent directory, overwriting the python scripts in the parent directory.

```
mv model.py ../
mv train_val.py ../
mv test.py ../
```

Then follow the README in parent folder. The analysis scripts may not work with this codebase.
