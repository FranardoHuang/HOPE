# EXP-GATE3-PLANNER-POLICY-RELEASE-BUILD — exact planner-policy portable Release

- 状态：completed
- 当前环节：portable Release source/binary gate 已通过；runtime gates 未运行
- 阶段/轴：部署验证线 / planner-policy 集成
- 人类负责人：franco
- 执行者：Codex
- 最高证据等级：E1
- 决定：adopt 这组 exact 源码进入 main 集成候选；只有实际合入 `main` 后才成为共享能力

This record is the detailed source of truth for the final integrated planner-policy candidate's
native C++ build attempt. Terms such as Gate3, planner, policy and exactness are defined in
[DEFINITIONS.md](../../DEFINITIONS.md). Here, **portable Release** means an optimized Linux C++ build
with Robot Operating System 2 (ROS 2) and the AimRT backend explicitly disabled; it is a
source/binary gate, not a simulator, backend or robot test.

## Question and decision rule

Question: can exact source commit `c0a8e46b0c0bec4d89040728a7fa64f064090432` configure, compile
both the native test binary (`run_tests`) and production ping-pong runner
(`a3_deploy_onnx_ref_pingpong`), pass the planner/first-tick focused tests and pass the complete
native test binary on Ubuntu 24.04 with GNU Compiler Collection (GCC) 13 Release semantics?

Decision rule:

- only an exact clean checkout, successful target generation/link, focused green result, complete
  native green result, strict finite-math flag audit and content-addressed binaries permit merge;
- a configure failure is an environment blocker, not a source pass or source failure;
- this build never starts the runner, backend, simulator, publisher or a real-robot command.

## Fixed environment

- Pod: Pod2, host `08396d34ac26`, Ubuntu `24.04.3`, kernel `6.8.0-101-generic`.
- Compiler: GCC/G++ `13.3.0`; GNU ld `2.42`; CMake `3.28.3`.
- Isolated root:
  `/workspace/codexschema/external_verify_c0a8e46b_20260712T161802Z_234469`.
- Source: detached, clean exact `c0a8e46b0c0bec4d89040728a7fa64f064090432`.
- ROS 2 Jazzy: absent. `HAS_ROS2=0`, `ENABLE_A3_ROS_MSGS=OFF`,
  `ENABLE_A3_AIMRT_BACKEND=OFF`, `ENABLE_TRT_INFERENCE=OFF`.
- Live training and evaluation checkouts were read-only and remained clean at
  `6d93bcb16c422a2f42748c2dc99432559653480b` and
  `46a0ce24524fdb843e55fe82ba4c045f2adc090f` respectively.

## Content-addressed dependency acquisition

No system package was installed and no train/eval file was used as a dependency source.

1. ONNX Runtime `1.19.2` was downloaded from the official GitHub release URL
   `https://github.com/microsoft/onnxruntime/releases/download/v1.19.2/onnxruntime-linux-x64-1.19.2.tgz`.
   The archive is `6,083,273` bytes with SHA-256
   `eb00c64e0041f719913c4080e0fed7d9963dc3aa9b54664df6036d8308dbcd33`. All tar entries were
   confined to one expected top-level directory. The 24-file extracted manifest has SHA-256
   `91377c885ff93fb940591a1c6a8463395c990ecfef4f0c2b2b86466c107a786c`.
2. The active Unitree software development kit (SDK) directory was initially absent. The only
   allowed local candidate was the same clone's tracked legacy subset at
   `agi/code_deployment/a3_deploy_example/thirdparty/unitree_sdk2`, Git tree
   `01b3e8903900cccb5c496cecb19fd6a51828327f`. Source and ignored destination had the same
   normalized tar SHA-256
   `d00d42fbc4a0762696d078d749516f89a3bb185184fc97ee730c2c1f530ee498` after copying. This proves
   copy fidelity only; the tracked subset is not the full vendor SDK.
3. Ubuntu noble's official `libmsgpack-cxx-dev=6.1.0-1build1` package was downloaded but not
   installed from
   `http://archive.ubuntu.com/ubuntu/pool/universe/m/msgpack-cxx/libmsgpack-cxx-dev_6.1.0-1build1_amd64.deb`.
   Its `.deb` SHA-256 is
   `4cb6277be55db29f4f04eb290fed2ad1ee7d9658ece9492cca2ce7ae05ce5538`; the declared Advanced
   Package Tool (APT) SHA-512 matched. The 738-file extracted manifest SHA-256 is
   `1d490eea4c6075570522cf6a94c0618304fa4c48a5b408b6bd2f2a9597f02e8f`.
4. Its declared Boost dependency was closed with fixed official noble packages only, again without
   installation: `libboost-dev=1.83.0.1ubuntu2` from
   `http://archive.ubuntu.com/ubuntu/pool/main/b/boost-defaults/libboost-dev_1.83.0.1ubuntu2_amd64.deb`
   (`.deb` SHA-256
   `6ccd980dde9960f27904d3b8a5ef04b419d7d372f7ca67b2a42083f03950f4be`) and
   `libboost1.83-dev=1.83.0-2.1ubuntu3.2` from
   `http://archive.ubuntu.com/ubuntu/pool/main/b/boost1.83/libboost1.83-dev_1.83.0-2.1ubuntu3.2_amd64.deb`
   (`.deb` SHA-256
   `519ecf2c64308527e15b6582955681d192a832250baa4bc424967aaf7d02d68f`). The concrete package's
   15,663-entry extracted manifest SHA-256 is
   `249d5eddbdc506c0c27bccad2f0ed9ad4be721b2f91e602a8882a439552cd10e`.
5. The complete private Unitree SDK was restored from the documented local handoff at
   `/Users/Franco/Dropbox/乒乓/nohope/vendor_assets/agibot/a3_deploy_example_full/thirdparty/unitree_sdk2`.
   A canonical manifest covered 863 directories, files and symlinks with SHA-256
   `e8f808e92b9b73cbcde2803b34ef48bb329941e40cd20f7b73282a1195588c13`. The local source and
   Pod2 no-clobber staging manifests were byte-identical before promotion. The x86_64 static
   library is `27,351,696` bytes with SHA-256
   `93ebabb2eca346892f23b9f78ece974a48091b44d745053e6911d3e294f74ec7`; the old incomplete copy
   remains preserved under a separate ignored name.
6. ZeroMQ (ZMQ) used the already-installed Ubuntu runtime `libzmq.so.5` (SHA-256
   `c56acd8baaec3d869f7e23a194870302acef6d86384f1f0fe946fb81c7015a42`) plus isolated official
   noble `libzmq3-dev=4.3.5-1build2` (`.deb` SHA-256
   `51d4da55be33741d4e74ca4724accfcf8c1f035167e338420b12fcf3cb45f62b`) and
   `cppzmq-dev=4.10.0-1build1` (`.deb` SHA-256
   `81375971369a7856a6007a8f4a4b84ecf59a1c03aadb391b204a9100a3909173`). No package was
   installed.
7. Eigen used official noble `libeigen3-dev=3.4.0-4build0.1`, `.deb` SHA-256
   `56c05756bd5bf1e7d1c30ea545786a51684b3c796f8289b20cc803b2f1e6f754`; its 537-entry
   extracted manifest SHA-256 is
   `3f77f22dd32a0e6c45726bdada650f6ca70523f80a6864f148663c269b97a3a9`.
8. YAML used matched official noble `libyaml-cpp0.8` and `libyaml-cpp-dev`, `.deb` SHA-256 values
   `2be57b812a1b082011d1b9bf6140fb875f19e153730be1ba965185b890b387f5` and
   `1b5a9ef5de12f24a6f99168f054346614a7a6060934c47067a9495cb239a8f51`. Their combined
   50-entry sysroot manifest SHA-256 is
   `0d4fe7ed862ad16d67efecb204133851bade62dcfa1876b1c59efd4517866e6d`.
9. The tests compiled the repo-declared official GoogleTest FetchContent source from
   `https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip`. The `1,090,859`-byte
   archive SHA-256 is `1f357c27ca988c3f7c6b4bf68a9395005ac6761f034046e9dde0896e3aba00e4`;
   its 276-entry extracted-source manifest SHA-256 is
   `2a4f595bfb053cdfc45b51a8d3445688fe7a710cc5409a8bf615d07da18c2464`. Matched official
   noble `googletest` and `libgtest-dev` version `1.14.0-1` supplied only the package config needed
   by the subsequent unconditional `find_package`; their `.deb` SHA-256 values are
   `196b763a2539c117ee2bbd5878ffa33b00ce8017ca6b63b89245b74c8889f1d1` and
   `c7dd1ecf6b816bf13ab6e1d8813d1ff0ae7a4fa09326a0cd467c3943d952117f`. The final link line
   binds the FetchContent-built `libgtest.a`/`libgtest_main.a`, not the Ubuntu static libraries.
10. JavaScript Object Notation (JSON) used official noble `nlohmann-json3-dev=3.11.3-1`, `.deb`
    SHA-256
    `85e4e95dc4bdc6034d593e1933ee88e085319fee36cae83c178e8ddc1a66e8fd`; its 478-entry
    manifest SHA-256 is `be6dc63b944aa80f30f2cf4e7d67b87f41217aec59d928f1f6efe5592b334d67`.

Every Ubuntu archive came from `archive.ubuntu.com` noble/noble-updates, matched the SHA-512
declared by APT and was extracted under the ignored isolated sysroot. The additional direct package
URIs were:

| Package | Official Ubuntu URI |
| --- | --- |
| `libzmq3-dev=4.3.5-1build2` | `http://archive.ubuntu.com/ubuntu/pool/universe/z/zeromq3/libzmq3-dev_4.3.5-1build2_amd64.deb` |
| `cppzmq-dev=4.10.0-1build1` | `http://archive.ubuntu.com/ubuntu/pool/universe/c/cppzmq/cppzmq-dev_4.10.0-1build1_amd64.deb` |
| `libeigen3-dev=3.4.0-4build0.1` | `http://archive.ubuntu.com/ubuntu/pool/universe/e/eigen3/libeigen3-dev_3.4.0-4build0.1_all.deb` |
| `libyaml-cpp0.8=0.8.0+dfsg-6build1` | `http://archive.ubuntu.com/ubuntu/pool/main/y/yaml-cpp/libyaml-cpp0.8_0.8.0%2bdfsg-6build1_amd64.deb` |
| `libyaml-cpp-dev=0.8.0+dfsg-6build1` | `http://archive.ubuntu.com/ubuntu/pool/main/y/yaml-cpp/libyaml-cpp-dev_0.8.0%2bdfsg-6build1_amd64.deb` |
| `googletest=1.14.0-1` | `http://archive.ubuntu.com/ubuntu/pool/universe/g/googletest/googletest_1.14.0-1_all.deb` |
| `libgtest-dev=1.14.0-1` | `http://archive.ubuntu.com/ubuntu/pool/universe/g/googletest/libgtest-dev_1.14.0-1_amd64.deb` |
| `nlohmann-json3-dev=3.11.3-1` | `http://archive.ubuntu.com/ubuntu/pool/universe/n/nlohmann-json3/nlohmann-json3-dev_3.11.3-1_all.deb` |

Exact SHA-512 values are in the acquisition logs.

## Attempts and accepted result

Fresh no-clobber build directories preserved every intermediate environment failure: missing
Unitree, MessagePack, ZMQ, Eigen, YAML, GTest and JSON; the isolated YAML package also required its
include/library prefix to be explicit because the repo links bare `yaml-cpp`. None of these attempts
changed source bytes. The ninth isolated build attempt (`r9`) was the first complete dependency
closure and is the accepted result.

The exact `r9` outcome is:

- both `run_tests` and `a3_deploy_onnx_ref_pingpong` compiled and linked in Release mode;
- `compile_commands.json` contains 80 commands; all 80 contain `-O3`, `-fno-fast-math` and
  `-fno-finite-math-only`; forbidden positive fast/finite-only math flags are absent;
- focused filter `PpPlannerInput.*:PpFirstTickJson.*`: **40/40 passed**;
- complete native suite: **238 total = 233 passed + 5 optional-asset skips + 0 failed**;
- skips were the three pre-existing CSV/observation fixtures, FK reference assets and the optional
  real-ONNX loader fixture. No safety/planner/first-tick test skipped;
- dependency closure was complete under the bound ignored-library paths;
- source, live training and evaluation checkouts remained clean at their exact recorded commits.

| Artifact | SHA-256 | Bytes / identity |
| --- | --- | --- |
| `run_tests` | `89cb57da63eae58680c9c978b95d28103177af20cde23779f269e6c452c16921` | `3,565,840`; build ID `4936e2cfd590b0a3355c9c926bc8cdebc1a32c0c` |
| `a3_deploy_onnx_ref_pingpong` | `c89856f4c440be0b424abc30bde186b784a6b99546aa2f08c37e1fee32f376a9` | `674,856`; build ID `2d137b869488d1861bab0ee96549f249751c539c` |
| `compile_commands.json` | `aa9e2290e76e4bb4a416265d22f92a74f177073c1ebc1a6665b85804c41a3b80` | 80 compile commands |

The content-addressed logs remain under the isolated root's `logs/` directory:

| Log | SHA-256 |
| --- | --- |
| `portable_release_exact_c0a8e46_20260713_r9_build.log` | `19fa3195caaac8236ab0e0f198c82384d9ef093d8a9123baf377fae5e347b969` |
| `portable_release_exact_c0a8e46_20260713_r9_identity.txt` | `982c75c0a7412135539c813ad7ce3474182abfb15f328bcb85c1dc52c2c3761f` |
| `portable_release_exact_c0a8e46_20260713_r9_audit.log` | `9faf1a366e31ba8bc2a61586b7c910572ddc219c95ae5b879f6eec8b2edd1efa` |
| `portable_release_exact_c0a8e46_20260713_r9_focused.log` | `17f1fd27bbb494bcda48dda03e5be5bf34cd122c5ca3c8b6a83618614c6c510f` |
| `portable_release_exact_c0a8e46_20260713_r9_full_native.log` | `d50fc6dba03028b0cc9455ac2ec61abbcc9d21f9d38446b4f01d6de690100adf` |
| `portable_release_exact_c0a8e46_20260713_r9_gtest_supplement.txt` | `5ceb479b391c986db0ce6981629c502fe5abfebb6417282e6fc9a9e07d5b7bde` |

The canonical result ledger has SHA-256
`045cc3cefac3b9f9addaa37c2868eea3ed6bdb41f9c6a8aee246dd1dbbeac6b6`; the no-clobber GoogleTest
source/link supplement has SHA-256
`5ceb479b391c986db0ce6981629c502fe5abfebb6417282e6fc9a9e07d5b7bde`.

## Latest-main integration and local regression

The main-integration worktree started from exact `origin/main`
`c68d40fd0550196bbf33410fcfbda66498886af2`. Twenty source/config/test paths differed from that
base and three already matched. All 23 effective paths changed between the planner feature base
`0239ecb` and validated candidate `c0a8e46` were restored byte-for-byte from `c0a8e46`; no source
conflict or semantic merge was performed. Unrelated latest-main motion and q50-supervisor files
were retained.

The sorted manifest format is one line per effective path,
`<file SHA-256><two spaces><repo-relative path>`. Its 23-path SHA-256 is
`8af1a2fc37dc912f41cb5609a687b481fbadbddc531ff4f430d6294796665fd3`. Both embedded source
contracts were then recomputed against the worktree: first-tick `20/20` and serve preregistration
`18/18` matched. Key exact file hashes are:

| File | SHA-256 |
| --- | --- |
| `pp_planner_input.hpp` | `1bb41836c532167ecf561c85d3c913d5ed6be590840214dfa747d5ace3c2eb2b` |
| `pp_policy.hpp` | `f91b1f5db4e35163abce6923567365bb177c094573de2740f443e3ada1b01de1` |
| `pp_reference_clock.hpp` | `d943b2f0f945a7a29ab734bd10d186b896ca2152ba658aecfc563951e3fcadfb` |
| `a3_pingpong_main.cpp` | `fc4ac19c57ccd3bbc7853785ec01a1d4c3b66666131de89f0f932c8c4f08b389` |
| planner `node.py` | `52a70da3fbcefbd4365d87b3e31ffa6ef11e7a7dfe630b1d1a6bf9b07dcea75d` |
| planner runtime contract | `a1f367a19e1ace7584cde030d9b315a3a549aac2ecbf6aad8033eea6930f4d1e` |
| flat command wire | `052384884382fbaffeaf2263121956cf030cca6bb12791fa15ced3142e6a21b9` |
| arena planner config | `12e8c8dcd937c208fad28dcfe5d820aef9ad53b1edd32490c9d2c89981792817` |
| simulator planner config | `efdb726e246fc4264a90ef6f187b19b891763fd5f1827706501708f4d3a19cea` |

Local tests on that exact source content passed:

- planner/source: `180 passed, 2 optional skips`;
- serve preregistration: `39 passed`;
- dependency-light first-tick state/JSON checks: `7 passed`;
- full root `tests/`: `521 passed, 9 optional skips`;
- design-only serve validator: exit 0 with all 49 runtime bindings explicitly blocked;
- launch-check negative control: exit 1 with exactly 49 `MISSING` bindings.

These host tests add latest-main compatibility evidence; they do not replace or transfer a new
portable Release binary result. Reusing the Pod2 `r9` result is valid only because every effective
source byte remains exactly the validated c0 content.

## Interpretation and remaining gates

This closes the exact integrated candidate's **portable Release source/binary gate** and removes
the earlier build-only NO-MERGE blocker. It does not claim ROS 2/AimRT Release closure, a formal
ONNX loader run, a planner-policy first backend tick, vendor MuJoCo behavior, continuous stability
or real hardware. The production runner binary was never executed; backend, simulator, publisher
and robot-command flags in the result ledger are all false.

Main integration preserves the validated source bytes. Any later source conflict or semantic
change invalidates this binary evidence and requires a new Release build. The next runtime step
remains the separately reviewed no-publish first-tick harness, not an unowned launch of the
production runner.
