# 단일 서버 다중 노드(VM 기반) Kubernetes 클러스터 확장 계획

## 목표
현재 단일 노드(`oran-server`) 위에 KVM/libvirt 기반 VM을 추가하여 K8s 다중 노드 토폴로지를 구성한다. Non-RT RIC + SMO 경량 컴포넌트를 별도 워커 노드에 배치할 수 있도록 한다.

## 사전 환경 점검 결과
- [x] CPU 가상화 지원 확인 (vmx/svm 48코어)
- [x] 메모리 여유 확인 (188Gi 중 156Gi available)
- [x] 디스크 여유 확인 (`/` 723G 사용 가능)
- [x] libvirt 미설치 → 설치 필요
- [x] 기존 K8s 클러스터(kubeadm v1.32.11) 단일 노드 동작 중

---

## ⚠ 핵심 결정 사항 (실행 전 확정 필요)

### 결정 1: control-plane 분리 전략
**Plan A (권장 ⭐, 비파괴적)**
- 호스트는 그대로 control-plane + AI/ML worker 겸임 유지
- VM은 ric/smo worker 1대만 추가
- 장점: 기존 워크로드 무중단, 87일치 데이터/상태 그대로 보존
- 단점: control-plane이 AI/ML과 같이 떠 있으므로 etcd 보호 효과는 부분적 (VM cgroup 격리만큼)

**Plan B (엄격한 옵션 B 토폴로지)**
- VM-cp(신규)를 별도 control-plane으로 만들고 호스트를 worker로 강등
- 기존 kubeadm 클러스터 재구성 필요 → **모든 워크로드 재배포(다운타임 발생)**
- 장점: 진정한 control-plane 분리
- 단점: 위험·시간 많이 듦

**→ 권장: Plan A로 시작. 운영 중 etcd 부하가 실제로 문제되면 그때 Plan B로 전환.**

### 결정 2: VM 갯수
- **2-node 안 (권장 ⭐)**: 호스트(cp+aiml) + VM-ric-smo
- 3-node 안: 호스트(cp+aiml) + VM-nonrtric + VM-smo (추후 확장 시)

**→ 권장: 2-node로 시작. SMO 컴포넌트 늘어나면 VM 추가.**

---

## 구성안 (Plan A + 2-node)

| 노드 | 종류 | 사양 | 역할 | 라벨 |
|---|---|---|---|---|
| `oran-server` (현재) | bare-metal | 48C / 188GB / GPU 2 | control-plane + AI/ML | `workload=aiml`, `gpu=nvidia` |
| `oran-ric-smo` (신규) | KVM VM | 12 vCPU / 48GB / 200GB | RIC/SMO worker | `workload=ric-smo` |

**자원 분배 검토**
- 호스트 가용: 156GB → VM에 48GB 할당 시 호스트는 ~108GB 여유 (충분)
- CPU: 48C → VM 12C, 호스트 36C (여전히 충분)
- 디스크: 723GB 여유 → VM 200GB 차지 (여유 523GB)

---

## 단계별 작업 목록

### Phase 1: 가상화 기반 설치 (완료 ✅)
- [x] 1.1 KVM/libvirt/virt-manager 패키지 설치
  - `qemu-kvm`, `libvirt-daemon-system`, `libvirt-clients`, `bridge-utils`, `virtinst`, `cloud-image-utils`, `cpu-checker`
- [x] 1.2 libvirtd 서비스 활성화 (active, enabled / oran 사용자 libvirt 그룹 등록 확인)
- [x] 1.3 가상화 동작 확인 (`virsh 10.0.0`, KVM acceleration available)

### Phase 2: VM 네트워크 구성 (완료 ✅)
- [x] 2.1 **`default` 네트워크 미사용** — Calico IP 풀(`192.168.0.0/16`)과 192.168.122.x가 겹쳐 충돌 위험. 자동시작 비활성화.
- [x] 2.1' **`oran-vm` 신규 네트워크 생성** — `10.10.0.0/24` 대역, 브리지 `oranbr0`, NAT 모드, 자동시작 등록.
  - VM에 IP 예약: `oran-ric-smo` → `10.10.0.10` (MAC `52:54:00:6a:1c:01`)
- [x] 2.2 충돌·도달성 검증
  - 호스트 인터페이스/Calico 풀과 충돌 없음 확인 (10.10.0.0/24는 클린)
  - iptables MASQUERADE/FORWARD 규칙 자동 적용 확인
  - kube-apiserver `*:6443` 바인딩 확인 → VM이 NAT 통해 `210.123.36.94:6443` 도달 가능
- [x] 부가: `fs.inotify.max_user_instances` 128→1024, `max_user_watches` 65536→524288 상향 (`/etc/sysctl.d/90-inotify.conf` 영구 반영) — dnsmasq 기동을 막던 한도 문제 해소

### Phase 3: VM 생성 (완료 ✅)
- [x] 3.1 Ubuntu 24.04 cloud image 다운로드 (`noble-server-cloudimg-amd64.img`, 601MB)
- [x] 3.2 cloud-init user-data 작성: 호스트네임 `oran-ric-smo`, 사용자 `oran-ric-smo` (sudo, NOPASSWD), 호스트 ed25519 SSH 키 주입, swapoff, kernel modules(`br_netfilter`,`overlay`), sysctl(`net.bridge.bridge-nf-call-iptables=1`, `net.ipv4.ip_forward=1`), 디스크 자동 확장(growpart/resize2fs)
- [x] 3.3 `virt-install` 실행 — 12 vCPU / 48GB RAM / 200GB qcow2 / `oran-vm` 네트워크 / cpu host-passthrough
- [x] 3.4 부팅·검증 완료
  - DHCP 예약대로 IP `10.10.0.10` 할당
  - SSH 접속 성공 (`oran-ric-smo@10.10.0.10`)
  - 디스크 200GB 정상 확장 (193G 가용)
  - Memory 47Gi, CPU 12, cloud-init `status: done`, swap off, 커널 모듈 로드됨
  - K8s API `210.123.36.94:6443/healthz` → `ok` (NAT 통한 도달 검증)
  - 인터넷 outbound OK

### Phase 4: VM에 K8s 노드 prerequisites 설치 (완료 ✅)
- [x] 4.1 swap 비활성화, 커널 모듈(br_netfilter, overlay) — cloud-init으로 사전 적용
- [x] 4.2 sysctl 파라미터 — cloud-init으로 사전 적용
- [x] 4.3 **containerd 1.7.28 설치** (`1.7.28-2~ubuntu.24.04~noble`, Docker 공식 저장소) + `SystemdCgroup=true` + `sandbox_image=registry.k8s.io/pause:3.10` 설정 + 자동시작
- [x] 4.4 **kubelet/kubeadm/kubectl 1.32.8 설치** (pkgs.k8s.io v1.32 저장소, 호스트와 동일 버전) + `apt-mark hold` 적용
- [x] 4.5 kubelet `enabled`(자동시작), join 전이므로 `inactive` 상태가 정상 — `crictl` runtime ping 정상

### Phase 5: 클러스터 join (완료 ✅)
- [x] 5.1 join 토큰 생성 (`kubeadm token create --print-join-command`)
- [x] 5.2 VM에서 `kubeadm join 210.123.36.94:6443` 성공 — TLS bootstrap 정상
- [x] 5.3 `oran-ric-smo` 노드 **Ready** (v1.32.8 / 12C / 49GB / 200GB)
- [x] 5.4 신규 노드에 `calico-node`, `kube-proxy`, `nvidia-device-plugin-daemonset` 자동 배포 (모두 Running)
- 비고: VM에 GPU가 없는데 `nvidia-device-plugin`이 자동 배포됨 → Phase 6에서 nodeSelector로 GPU 노드에만 한정시킬 것

### Phase 6: 노드 라벨링 & taint (완료 ✅)
- [x] 6.1 `oran-server`에 `workload=aiml`, `gpu=nvidia` 라벨 적용
- [x] 6.2 `oran-ric-smo`에 `workload=ric-smo` 라벨 적용
- [x] 6.2' **`nvidia-device-plugin-daemonset`에 `nodeSelector: gpu=nvidia` 패치** — VM(GPU 없음)에 불필요하게 떠있던 파드 제거
- [x] 6.3 **taint 미적용** (사용자 결정) — nodeSelector만으로 스케줄링 제어
- [x] 6.4 검증: 기존 50개 Pod 재스케줄링 없음, `nodeSelector: workload=ric-smo` 테스트 Pod가 정확히 신규 노드에 배치됨

### Phase 7: 보조 컴포넌트 도입 (완료 ✅)
- [x] 7.1 **metrics-server** 설치 (`--kubelet-insecure-tls` 추가) — `kubectl top` 정상 동작
- [x] 7.2 **MetalLB v0.14.8** 설치 + IPAddressPool `lan-pool` (`210.123.36.200-210`) + L2Advertisement (인터페이스 `eno1`)
  - speaker DaemonSet에 `nodeSelector: workload=aiml` 추가 (LAN 인터페이스 없는 VM에서 ARP 못 함)
  - **`istio-ingressgateway` EXTERNAL-IP `210.123.36.200` 할당 ✅** (84일간 pending 해소)
- [x] 7.3 **kube-prometheus-stack** 설치 (네임스페이스 `monitoring`) — Prometheus / Alertmanager / Grafana / kube-state-metrics / node-exporter (DaemonSet)
  - 워크로드는 `nodeSelector: workload=ric-smo`로 VM에 배치
  - Grafana NodePort `32030` (admin 비밀번호: `oran-admin`)
  - Prometheus 7일 보존, 20Gi PVC / Grafana 5Gi PVC / Alertmanager 5Gi PVC

### NFS 인프라 수정 (Phase 7 도중 발견된 이슈 → 해결 ✅)
- [x] 호스트 `/etc/exports`에 `10.10.0.0/24` 대역 추가 — VM 노드도 NFS 접근 가능
- [x] VM에 `nfs-common` 설치 — mount.nfs 헬퍼
- [x] `nfs-subdir-external-provisioner` 패치
  - `NFS_SERVER` env: `localhost` → `10.10.0.1` (양 노드에서 도달 가능한 브리지 IP)
  - 파드 자체의 NFS volume `server` 필드도 `10.10.0.1`로 수정
  - `nodeSelector: workload=aiml` 추가 (NFS 서버는 호스트의 로컬 디렉터리이므로 호스트 노드에 고정)
- 비고: 기존 PVC들(cassandra, postgres, mysql 등)은 `localhost`로 이미 마운트되어 있고 호스트 노드에서 정상 동작 중 — 영향 없음. 신규 PVC만 새 server 주소로 생성됨.

### Phase 8: 검증 (완료 ✅)
- [x] 8.1 nodeSelector 스케줄링 정상 (`oran-server`: 52 파드 / `oran-ric-smo`: 10 파드 — calico/proxy/metrics-server/metallb-controller/monitoring 스택)
- [x] 8.2 KServe InferenceService 8개 중 6개 Ready=True / 2개(transformer-cu, transformer-du)는 **사전 이슈**(2026-04-14부터 TensorFlow 호환성 문제) — 이번 작업과 무관
- [x] 8.3 Kubeflow 15개 파드 모두 Running, traininghost 8개 모두 Running, rapp-test Running, cert-manager/istio-system/kserve/knative-serving 이상 없음
- [x] 8.4 Cassandra `nodetool status` 정상, PostgreSQL/InfluxDB 파드 Running

### Phase 9: Non-RT RIC + SMO 설치 (완료 ✅)
- [x] 9.1 Prerequisites — `yq`(snap), `helm cm-push` 플러그인 설치
- [x] 9.2 dep 리포지토리 클론 (`--recursive`로 `ric-dep`, `smo-install/multicloud-k8s`, `smo-install/onap_oom` 서브모듈 포함)
- [x] 9.3 컴포넌트 선택 + 사전 준비
  - 사용자 결정: **Kong** ingress 사용, **Full** 컴포넌트(doc 그대로)
  - `smo-storage` StorageClass 생성 (`nfs-subdir-external-provisioner` 재사용, override가 참조)
  - `oran-override.yaml` 수정: `installNonrtricgateway: false` (이후 9.6에서 다시 true로)
- [x] 9.4 ChartMuseum + Helm 환경 설정
  - 기존 AIMLFW용 ChartMuseum(`:8879`)은 그대로 유지
  - SMO 전용 ChartMuseum(`:18080`) 신규 기동 (`0-setup-charts-museum.sh`)
  - `local` helm repo 재등록 (8879 → 18080)
  - `0-setup-helm3.sh`로 deploy/undeploy 플러그인 + 외부 repo(oran/strimzi/openebs/mariadb) 등록
- [x] 9.5 Helm 차트 빌드 — `1-build-all-charts.sh` (153개 차트 ChartMuseum에 푸시)
- [x] 9.6 NodePort 32080 충돌 처리 — Kong proxy NodePort `32080` → **`32180`**으로 변경 (override yaml에서 중첩 차트 경로 `kong.kong.proxy.http.nodePort` 적용. LeoFS는 그대로)
- [x] 9.7 `2-install-oran.sh` 실행 — ONAP umbrella + 11 서브차트 + Non-RT RIC + SMO 배포
- [x] 9.8 검증 — `onap` 28/28, `nonrtric` 20/20, `smo` 27/27, `strimzi-system` 1/1 모두 Ready

### Phase 9 도중 발견된 이슈 → 해결 ✅
- [x] **Kong NodePort 32180 override 미적용** — kong이 중첩 차트(wrapper kong → upstream kong 2.38.0) 구조라 경로가 `kong.proxy.http.nodePort`가 아닌 `kong.kong.proxy.http.nodePort`. 수정 후 재설치
- [x] **controlpanel이 nonrtricgateway 호출하다 CrashLoop** — doc은 "둘 중 하나만"이라 했지만 controlpanel이 nonrtricgateway에 의존. 두 게이트웨이 공존이 정상이라 다시 활성화(`installNonrtricgateway: true`)
- [x] **Strimzi 자동생성 시크릿 미복사** (재설치 시 install-*.sh의 secret-copy 단계 누락) — `dmeparticipant-ku`(nonrtric), `topology-*-ku`/`focom-*-ku`/`ncmp-*-ku`/`redpanda-console-ku`(smo) 6개 수동 복사
- [x] **JDK 17.0.2 + cgroup v2 NPE** (JDK-8281561) — `informationservice`, `dmaapadapterservice`, `nonrtricgateway`, `pm-producer-json2kafka` 4개 파드에 `JAVA_TOOL_OPTIONS=-Dspring.autoconfigure.exclude=...MetricsAutoConfiguration`(System/Tomcat/Jvm/Logback/WebMvc 메트릭 자동설정 제외) 적용으로 우회

### Post-Phase 9: Grafana LoadBalancer 노출 (완료 ✅)
- [x] `kps-grafana` 서비스 NodePort → **LoadBalancer**로 전환, MetalLB가 **`210.123.36.202`** 할당 (`metallb.universe.tf/loadBalancerIPs` annotation 명시)
- [x] values 파일을 `/tmp/kps-values.yaml` → **`/usr/local/o-ran/kps-values.yaml`** 영구 위치로 이동
- [x] `helm upgrade kps -f /usr/local/o-ran/kps-values.yaml --reuse-values=false` 실행
- [x] 검증: `curl http://210.123.36.202/api/health` → `{"database":"ok","version":"13.0.1"}`

---

## 위험 및 완화책

| 위험 | 영향 | 완화책 |
|---|---|---|
| libvirt 설치 중 패키지 충돌 | 호스트 네트워크 영향 가능성 | 설치 전 `apt list --installed`로 기존 패키지 확인, 단계적 설치 |
| VM 네트워크가 Calico CNI와 충돌 | 클러스터 통신 두절 | libvirt default NAT(`192.168.122.0/24`)는 Pod CIDR(`192.168.17.0/24`)과 다른 대역 사용 — 충돌 없음 |
| `kubeadm join` 시 버전 불일치 | join 실패 | VM에 호스트와 동일한 v1.32.11 명시 설치 |
| VM 자원 과다 할당으로 호스트 OOM | 기존 워크로드 영향 | 보수적 할당(48GB), 호스트에 ~100GB 이상 여유 유지 |
| GPU 자원이 잘못된 노드로 스케줄 | 추론 실패 | GPU 라벨/`nvidia.com/gpu` resource는 호스트에만 존재 → VM에는 자연히 안 감 |

## 롤백 계획
- Phase 5 이전 단계는 VM만 삭제하면 원복 (`virsh destroy && virsh undefine`)
- Phase 5 이후 (join 완료 후) 롤백:
  - `kubectl drain oran-ric-smo --ignore-daemonsets`
  - `kubectl delete node oran-ric-smo`
  - VM에서 `kubeadm reset`
  - VM 삭제

## 확인 필요 사항 (사용자 결정 대기)
1. **Plan A vs Plan B?** (control-plane 분리 전략)
2. **2-node vs 3-node?** (시작 규모)
3. **VM 사양(12C/48GB/200GB) 그대로 진행?** 더 크게/작게 조정?
4. **VM 네트워크: libvirt NAT vs LAN bridge?** (NAT 권장)

---

## Review (Phase 1~9 완료, 2026-05-08)

### 최종 클러스터 토폴로지
| 노드 | 역할 | IP | 라벨 | CPU | RAM |
|---|---|---|---|---|---|
| `oran-server` | control-plane + AI/ML worker | 210.123.36.94 | `workload=aiml`, `gpu=nvidia` | 48C (사용 1.6C) | 188Gi (사용 75Gi, 38%) |
| `oran-ric-smo` | RIC/SMO worker (KVM VM) | 10.10.0.10 | `workload=ric-smo` | 12C (사용 0.26C) | 49Gi (사용 10.5Gi, 21%) |
- 클러스터 총 파드 **141개** (기존 AI/ML 50 + 모니터링/MetalLB 등 15 + ONAP/RIC/SMO 76)

### 추가된 인프라
- KVM/QEMU/libvirt 가상화 스택 (VM 1대 운영 중)
- libvirt 네트워크 `oran-vm` (10.10.0.0/24, NAT, 브리지 `oranbr0`)
- metrics-server (`kubectl top` 활성화)
- MetalLB v0.14.8 — LAN 풀 `210.123.36.200-210`, speaker는 `workload=aiml`로 한정
- kube-prometheus-stack — Prometheus / Grafana / Alertmanager / kube-state-metrics / node-exporter (DaemonSet)
- **ChartMuseum (`:18080`)** — SMO 전용 (기존 AIMLFW용 `:8879`와 별개)
- **ONAP** umbrella + 11 서브차트 (Strimzi, Mariadb-galera, Postgres, CPS, DCAEgen2-services, Policy, SDNC, repository-wrapper, roles-wrapper)
- **Non-RT RIC**: rAppManager, ServiceManager, PMS, ICS, ControlPanel, Topology, DmaapAdapter, DmeParticipant, CapifCore, Kong, A1 simulators 6개 (osc/std/std2 × 2)
- **SMO**: RANPM (kafka-producer-pm-* 시리즈, pm-producer-json2kafka, dfc, pmlog), TEIV (topology-exposure/ingestion + adapters), Redpanda Console
- **smo-storage** StorageClass (nfs-subdir-external-provisioner 재사용)

### 변경된 인프라
- `nfs-subdir-external-provisioner` — NFS_SERVER `localhost`→`10.10.0.1`, 호스트 노드에 고정
- `/etc/exports` — `10.10.0.0/24` 대역 허용 추가
- `nvidia-device-plugin-daemonset` — `nodeSelector: gpu=nvidia` 추가
- `istio-ingressgateway` — EXTERNAL-IP `<pending>` → `210.123.36.200` (84일 만에 해소)
- `fs.inotify.max_user_instances` 128→1024, `max_user_watches` 65536→524288 (영구 적용)
- `kps-grafana` — NodePort → LoadBalancer (`210.123.36.202`)

### 외부 접속 포트
| 서비스 | 접근 |
|---|---|
| Istio Ingress | `https://210.123.36.200` (LoadBalancer) ⭐ NEW |
| **Kong (Non-RT RIC API gateway)** | `http://210.123.36.201` (LoadBalancer) / NodePort `32180` ⭐ NEW |
| **Grafana** | `http://210.123.36.202` (LoadBalancer, admin/`oran-admin`) ⭐ NEW |
| Kong Admin | NodePort `32081` |
| nonrtricgateway | NodePort `30093` |
| 기존 NodePort (Kubeflow UI 등) | 변경 없음 |

### MetalLB IP 풀 사용 현황
- `210.123.36.200`: istio-ingressgateway
- `210.123.36.201`: oran-nonrtric-kong-proxy
- `210.123.36.202`: kps-grafana
- 가용: `210.123.36.203-210` (8개)

### 진행 중 발견 → 해결한 이슈 (전체)
1. Calico Pod CIDR(192.168.0.0/16)과 libvirt default 네트워크(192.168.122.0/24) **대역 충돌** → `default` 사용 안 하고 `oran-vm`(10.10.0.0/24) 신설
2. `dnsmasq` 기동 실패 (inotify 한도 부족) → kernel 한도 영구 상향
3. NFS provisioner가 `localhost` 사용 → VM 노드에서 마운트 불가 → provisioner 설정/exports/VM 패키지 모두 수정
4. nvidia-device-plugin이 GPU 없는 VM에 자동 배포 → nodeSelector로 한정
5. Kong NodePort 32080 충돌(LeoFS와) → 32180으로 변경 (중첩 차트 경로 `kong.kong.proxy.http.nodePort` 적용)
6. controlpanel의 nonrtricgateway 의존 → Kong과 공존시켜 해결
7. Strimzi 자동생성 시크릿 미복사(재설치 시) → 6개 시크릿 수동 복사
8. JDK 17.0.2 + cgroup v2 NPE (JDK-8281561) → 4개 Spring Boot 파드에 `JAVA_TOOL_OPTIONS` 메트릭 자동설정 제외로 우회

### 사후 권장 (후속 작업 후보)
- Grafana TLS / 비밀번호 강화 / OIDC(Keycloak) 연동
- Kiali 도입 (Non-RT RIC/SMO 서비스 메시 토폴로지 시각화)
- ric-smo 노드의 자원 사용률은 21%로 여유 있음 — Non-RT RIC 워크로드 일부를 ric-smo로 강제 이전(post-patch)하면 호스트 부하 분산 가능

---

## Baseline rApp 구현 (Ch4 실험 2) — `Inference_Service/ids-rapp-baseline-bi-lstm/`

설계 문서 `baseline_rApp_구현설계.docx` 그대로 초기 구현 완료. Proposed(`ids-rapp-bi-lstm`)를
in-process 임베드 방식으로 개조 + v3 Table 10 벤치 3종(latency·resource·retraining) 추가.

### 만든 것 (체크리스트)
- [x] `train/train.py` — 로컬 CSV 단일 학습(KFP/FeatureStore 제거, CV 생략·validation_split=0.2), `model.keras`+`scaler.pkl` 저장. FEATURES는 `rapp/app/features.py`에서 단일 출처 import.
- [x] `rapp/app/inference.py` — KServe HTTP 제거, in-process `tf.keras` lazy 로드. `predict_batch/predict_single` 시그니처 유지 + `model_loaded()`/`reload_model()`.
- [x] `rapp/app/preprocess.py` — Model.zip 다운로드 제거, `joblib.load(SCALER_PATH)`만.
- [x] `rapp/app/config.py` — KServe 항목 제거, `MODEL_PATH/SCALER_PATH` 추가, `WINDOW_SIZE=20`(학습 일치), `MODEL_NAME=baseline-bi-lstm`.
- [x] `rapp/app/bench.py` — ① `bench_latency`(방식 A 배치단위/방식 B 누적), ② `bench_resource`(1회 추론 중 CPU%·RSS), ③ `bench_retraining_availability`(재학습 전후 1초 간격 시계열).
- [x] `rapp/app/main.py` — `/bench/latency` `/bench/resource` `/bench/retraining` 추가, `/healthz`에 `model_loaded`(kserve_reachable 제거), `_RUN` phase 확장.
- [x] verbatim copy: `features.py`/`influx_client.py`/`metrics.py`/`krm/namespace.yaml` (Proposed와 byte-identical 확인).
- [x] `rapp/Dockerfile`(TF 2.17.0-gpu, model+train.py 베이크), `krm/service.yaml`·`deployment.yaml`(GPU 1·oran-server 핀·mem 4Gi), `rapp-image-build.sh`(model 산출물 체크), `.gitignore`(model/).

### 검증 (호스트에서 가능한 범위)
- py_compile 전체 통과, krm YAML 파싱 OK.
- `make_windows` 정합성(라벨=윈도우 마지막 시점) + FEATURES(20)/N_CLASSES(6) 단위 테스트 통과.
- TF 불요 모듈(config/features/metrics/preprocess/inference/bench) import OK.
- 스텁 모델로 `bench_latency`(A: n_measure 표본, B: batch_size 표본+cumulative)·`bench_resource`·retraining 가드 동작 + 결과 JSON 저장 확인.

### 설계 대비 의도적 차이 (기록)
1. **빌드 컨텍스트**: model/이 rapp/의 형제라 `COPY model /model`이 context=rapp에서 불가 → build context를 baseline 루트로, Dockerfile은 `COPY rapp/app`·`train/train.py`·`model`. (Dockerfile 인라인 주석은 Docker가 인자로 오해하므로 줄 분리)
2. **bench_latency `n_measure`**: 설계 시그니처엔 없으나 방식 A의 mean/p95/p99/std 산출에 반복이 필요 → `n_measure`(기본 100) 추가(이전 턴 사용자 결정 "배치당 반복 측정 유지" 반영). 방식 B는 단건×batch_size 호출이 그 자체로 표본을 만듦.

### 남은 환경 의존 검증 (TF/클러스터 필요 — 호스트 불가)
- `python train/train.py --epochs 5` 스모크 → `model/` 산출물 (호스트에 TF 미설치, GPU 컨테이너/학습환경에서 실행).
- 이미지 빌드/배포/`/healthz`·`/predict`·`/evaluate`·`/bench/*` curl 검증 (oom_ids InfluxDB + GPU 노드 필요).
- `/bench/retraining`은 컨테이너에 train CSV 마운트(`TRAIN_CSV=/data/...`) 필요 — 초기 deployment엔 데이터 볼륨 미포함(문서화).
