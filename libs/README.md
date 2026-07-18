# Bundled JavaScript libraries

`three.min.js` is Three.js **r128** (UMD build), vendored here so UniGuide runs
offline. The Qt host loads it with a file base-URL pointing at this folder.

To refresh it:

```bash
curl -L -o three.min.js https://raw.githubusercontent.com/mrdoob/three.js/r128/build/three.min.js
```
