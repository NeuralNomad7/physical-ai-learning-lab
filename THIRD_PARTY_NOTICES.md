# Third-party notices

## Stored in this repository

### MuJoCo Menagerie: Franka Emika Panda model

- **Location:** `assets/third_party/mujoco_menagerie/franka_emika_panda/`, 72 files:
  - `panda.xml`, `scene.xml`, `README.md`, `CHANGELOG.md` and `LICENSE`;
  - 67 mesh files under `assets/`.
- **Source:** <https://github.com/google-deepmind/mujoco_menagerie>, directory
  `franka_emika_panda`.
  - Pinned commit: `4d038b3feae26ec82b46a4d586379114012a8ac7` (2026-09-29).
  - The last commit that changed the directory is
    `71f066ad0be9cd271f7ed58c030243ef157af9f4` (2026-07-04).
  - Fetched 2026-10-02.
- **Licence:** Apache License 2.0. The full text is in `LICENSE` in that directory. The
  directory has no NOTICE file, and the licence text has no copyright line.
- **Origin:** according to its README, the model was developed by Franka Emika and is
  derived from the publicly available URDF description in
  [frankaemika/franka_ros](https://github.com/frankaemika/franka_ros/tree/develop/franka_description).
- **Modifications: none.**
  - The files are byte-identical to the pinned commit.
    `assets/third_party/mujoco_menagerie/franka_emika_panda.manifest.json` records the
    SHA-256 of each file, and `doctor` and the tests check them.
  - The project scene, `assets/scenes/panda_push_scene.xml`, includes `panda.xml` unchanged
    through MJCF `<model>` and `<attach>` (prefix `panda/`).
  - Menagerie's own `scene.xml` is kept with the model but not used.
- **Images:** the images in `docs/evidence/phase1/` are renders that show this model.

## Installed, not stored in this repository

`uv sync --locked` installs these Python packages from PyPI into `.venv/`, which is not
part of the repository (see `.gitignore`).

- The licences below are as declared in each package's installed metadata. "Classifier"
  marks packages that declare their licence only through a trove classifier.
- The full licence texts are in `.venv\Lib\site-packages\<package>-<version>.dist-info\licenses\`.
  pyopengl has no licence file there.

| Package | Version | Declared licence | Used for |
| --- | --- | --- | --- |
| mujoco | 3.14.0 | Apache-2.0 | Physics, viewer, offscreen rendering |
| numpy | 2.5.3 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 | Arrays and random number generation |
| pillow | 12.3.0 | MIT-CMU | PNG files, status panels, text on images |
| pytest | 9.1.1 | MIT | Tests (development dependency) |
| absl-py | 2.5.0 | Apache-2.0 | Dependency of mujoco |
| etils | 1.14.0 | Apache Software License (classifier) | Dependency of mujoco |
| fsspec | 2026.9.0 | BSD-3-Clause | Dependency of etils |
| glfw | 2.10.2 | MIT | Dependency of mujoco (viewer window) |
| pyopengl | 3.1.10 | BSD License (classifier) | Dependency of mujoco |
| typing_extensions | 4.16.0 | PSF-2.0 | Dependency of etils |
| websockets | 17.1 | BSD-3-Clause | Dependency of mujoco |
| zipp | 4.1.0 | MIT | Dependency of etils |
| colorama | 0.4.6 | BSD License (classifier) | Dependency of pytest |
| iniconfig | 2.3.0 | MIT | Dependency of pytest |
| packaging | 26.3 | Apache-2.0 OR BSD-2-Clause | Dependency of pytest |
| pluggy | 1.6.0 | MIT | Dependency of pytest |
| pygments | 2.21.0 | BSD-2-Clause | Dependency of pytest |

Some of these wheels contain further third-party components with their own notices:

- **mujoco:** components such as GLFW, Eigen, Abseil and Qhull. They are listed in
  `mujoco-3.14.0.dist-info\licenses\LICENSES_THIRD_PARTY.md`.
- **numpy:** further licence files in its `licenses` folder.
- **glfw:** the GLFW shared library and a Visual C++ runtime DLL.
- **pyopengl:** freeglut and GLE DLLs, with their licence files in `OpenGL\DLLS`.

### Font

The text in the viewer's status panels and in the rendered images uses Pillow's built-in
default font, `ImageFont.load_default`.

- It is a reduced version of Aileron Regular by Sora Sagano (dotcolon).
- Aileron is released under
  [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/); see
  <https://dotcolon.net/fonts/aileron/>.
- The font comes with Pillow and is not stored in this repository.

## This project

The project's own code, scene and documentation do not have a licence file yet. Choose
one before publishing or sharing the repository.
