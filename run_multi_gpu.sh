#!/bin/bash

################################################################################
# SAM3 Multi-GPU Distributed Execution Script
# 이미지를 GPU별 폴더로 분산 배치
################################################################################

# ========== 기본 설정 ==========
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="${SCRIPT_DIR}/sam3_offline.py"
MODEL_DIR="${SCRIPT_DIR}/models"
LOG_DIR="${SCRIPT_DIR}/logs"
WORK_DIR="${SCRIPT_DIR}/work"  # GPU별 이미지 분산 작업 디렉토리

# 디렉토리 생성
mkdir -p "${LOG_DIR}"
mkdir -p "${WORK_DIR}"

# ========== 사용자 설정 ==========
# GPU 리스트 (사용할 GPU 인덱스)
GPU_LIST=(0 1 2 3 4 5 6 7)

# 데이터 경로
IMAGE_DIR="/workspace/datasets/db2/101.etc/solbrain/data/JPEGImages"
LABEL_DIR="/workspace/datasets/db2/101.etc/solbrain/data/labels"
VIZ_DIR="/workspace/datasets/db2/101.etc/solbrain/data/result"

# 클래스 설정
CLASSES="person:0,fallen_person:1,head:2,safety helmet:3,car:6,bus:7,truck:8,forklift:9,chair:11"

# 추론 설정
THRESHOLD=0.3
CHUNK_SIZE=8

# 표시 옵션
SHOW_REALTIME=false
SAVE_VISUALIZATION=true

# 분산 방식 선택
# "symlink" : 심볼릭 링크 생성 (빠름, 저장공간 절약) ⭐ 추천
# "copy"    : 파일 복사 (느림, 저장공간 2배 필요)
DISTRIBUTE_METHOD="symlink"

# ========== 색상 코드 ==========
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# ========== 함수 정의 ==========

print_header() {
    echo -e "${CYAN}================================================================${NC}"
    echo -e "${CYAN}  $1${NC}"
    echo -e "${CYAN}================================================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

# GPU 개수 확인
check_gpu_count() {
    if ! command -v nvidia-smi &> /dev/null; then
        print_error "nvidia-smi를 찾을 수 없습니다."
        exit 1
    fi
    
    GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
    
    if [ $GPU_COUNT -eq 0 ]; then
        print_error "사용 가능한 GPU가 없습니다."
        exit 1
    fi
    
    print_success "감지된 GPU: ${GPU_COUNT}개"
}

# 이미지 분산 배치 (심볼릭 링크 또는 복사)
distribute_images() {
    local image_dir=$1
    local num_gpus=$2
    local work_dir=$3
    local method=$4
    
    print_info "이미지 파일 검색 중..."
    
    # 전체 이미지 리스트 생성
    local temp_list="${work_dir}/all_images.txt"
    find "${image_dir}" -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.bmp" \) > "${temp_list}"
    
    local total_images=$(wc -l < "${temp_list}")
    print_info "총 이미지: ${total_images}개"
    
    if [ $total_images -eq 0 ]; then
        print_error "이미지를 찾을 수 없습니다: ${image_dir}"
        exit 1
    fi
    
    # GPU별 디렉토리 생성
    print_info "GPU별 작업 디렉토리 생성 중..."
    for ((i=0; i<num_gpus; i++)); do
        local gpu_dir="${work_dir}/gpu_${i}"
        mkdir -p "${gpu_dir}"
    done
    
    # 이미지 분산
    print_info "이미지 분산 중 (방식: ${method})..."
    
    local line_num=0
    while IFS= read -r img_path; do
        local gpu_idx=$((line_num % num_gpus))
        local gpu_dir="${work_dir}/gpu_${gpu_idx}"
        local img_name=$(basename "${img_path}")
        
        if [ "${method}" == "symlink" ]; then
            # 심볼릭 링크 생성
            ln -sf "${img_path}" "${gpu_dir}/${img_name}"
        else
            # 파일 복사
            cp "${img_path}" "${gpu_dir}/${img_name}"
        fi
        
        line_num=$((line_num + 1))
        
        # 진행률 표시 (10000개마다)
        if [ $((line_num % 10000)) -eq 0 ]; then
            print_info "진행: ${line_num}/${total_images}"
        fi
    done < "${temp_list}"
    
    print_success "이미지 분산 완료!"
    
    # 각 GPU별 이미지 개수 확인
    echo ""
    for ((i=0; i<num_gpus; i++)); do
        local gpu_dir="${work_dir}/gpu_${i}"
        local count=$(find "${gpu_dir}" -type f -o -type l | wc -l)
        print_success "GPU ${i}: ${count}개 이미지 → ${gpu_dir}"
    done
    
    # 예상 처리 시간
    local images_per_gpu=$((total_images / num_gpus))
    local time_per_image=1  # 초
    local total_seconds=$((images_per_gpu * time_per_image))
    local hours=$((total_seconds / 3600))
    print_info "예상 처리 시간: 약 ${hours}시간 (GPU당 ${images_per_gpu}개)"
    
    rm "${temp_list}"
}

# GPU별 프로세스 실행
run_gpu_process() {
    local gpu_id=$1
    local gpu_image_dir=$2
    local log_file="${LOG_DIR}/gpu_${gpu_id}.log"
    
    print_info "GPU ${gpu_id} 시작..."
    
    local cmd="python3 ${PYTHON_SCRIPT}"
    cmd="${cmd} --gpu 0"  # CUDA_VISIBLE_DEVICES로 GPU 지정하므로 0 사용
    cmd="${cmd} --model_dir ${MODEL_DIR}"
    cmd="${cmd} --image_dir ${gpu_image_dir}"
    cmd="${cmd} --label_dir ${LABEL_DIR}"
    cmd="${cmd} --classes \"${CLASSES}\""
    cmd="${cmd} --threshold ${THRESHOLD}"
    cmd="${cmd} --chunk_size ${CHUNK_SIZE}"
    
    if [ "${SAVE_VISUALIZATION}" = true ]; then
        cmd="${cmd} --save_viz --viz_dir ${VIZ_DIR}"
    fi
    
    if [ "${SHOW_REALTIME}" = true ]; then
        cmd="${cmd} --show"
    fi
    
    echo "Starting: CUDA_VISIBLE_DEVICES=${gpu_id} ${cmd}" > "${log_file}"
    CUDA_VISIBLE_DEVICES=${gpu_id} eval "${cmd}" >> "${log_file}" 2>&1 &
    
    local pid=$!
    echo $pid > "${LOG_DIR}/gpu_${gpu_id}.pid"
    
    print_success "GPU ${gpu_id} PID: ${pid}"
}

# 프로세스 모니터링
monitor_processes() {
    local num_gpus=$1
    
    print_header "프로세스 모니터링 (10초마다 갱신)"
    
    local all_done=false
    while [ "$all_done" = false ]; do
        all_done=true
        
        echo ""
        echo "$(date '+%Y-%m-%d %H:%M:%S') - 상태"
        echo "----------------------------------------"
        
        for ((i=0; i<num_gpus; i++)); do
            local pid_file="${LOG_DIR}/gpu_${i}.pid"
            
            if [ -f "${pid_file}" ]; then
                local pid=$(cat "${pid_file}")
                
                if ps -p $pid > /dev/null 2>&1; then
                    echo -e "${GREEN}GPU ${i}: 실행중 (PID: ${pid})${NC}"
                    all_done=false
                else
                    echo -e "${YELLOW}GPU ${i}: 완료${NC}"
                fi
            fi
        done
        
        if [ "$all_done" = false ]; then
            sleep 10
        fi
    done
    
    print_success "모든 GPU 완료!"
}

# 결과 요약
print_summary() {
    print_header "실행 결과"
    
    for ((i=0; i<${#GPU_LIST[@]}; i++)); do
        local gpu_id=${GPU_LIST[$i]}
        local log_file="${LOG_DIR}/gpu_${gpu_id}.log"
        
        echo ""
        echo "GPU ${gpu_id}:"
        echo "----------------------------------------"
        
        if [ -f "${log_file}" ]; then
            tail -20 "${log_file}"
        fi
    done
    
    echo ""
    print_info "상세 로그: ${LOG_DIR}"
}

# 정리
cleanup() {
    print_warning "종료 중..."
    
    # 프로세스 종료
    for pid_file in "${LOG_DIR}"/*.pid; do
        if [ -f "${pid_file}" ]; then
            local pid=$(cat "${pid_file}")
            if ps -p $pid > /dev/null 2>&1; then
                kill -TERM $pid 2>/dev/null
                print_info "프로세스 종료: ${pid}"
            fi
            rm "${pid_file}"
        fi
    done
    
    # 작업 디렉토리 정리 (선택)
    read -p "작업 디렉토리를 삭제하시겠습니까? (y/N): " answer
    if [ "$answer" == "y" ] || [ "$answer" == "Y" ]; then
        print_info "작업 디렉토리 삭제 중..."
        rm -rf "${WORK_DIR}"
        print_success "삭제 완료"
    fi
    
    exit 0
}

# ========== 메인 ==========

main() {
    print_header "SAM3 Multi-GPU Image Processing"
    
    trap cleanup SIGINT SIGTERM
    
    # 환경 확인
    print_header "환경 확인"
    
    if [ ! -f "${PYTHON_SCRIPT}" ]; then
        print_error "스크립트 없음: ${PYTHON_SCRIPT}"
        exit 1
    fi
    print_success "스크립트: ${PYTHON_SCRIPT}"
    
    if [ ! -d "${MODEL_DIR}" ]; then
        print_error "모델 없음: ${MODEL_DIR}"
        exit 1
    fi
    print_success "모델: ${MODEL_DIR}"
    
    if [ ! -d "${IMAGE_DIR}" ]; then
        print_error "이미지 없음: ${IMAGE_DIR}"
        exit 1
    fi
    print_success "이미지: ${IMAGE_DIR}"
    
    check_gpu_count
    
    local num_gpus=${#GPU_LIST[@]}
    print_info "사용 GPU: ${GPU_LIST[*]} (${num_gpus}개)"
    print_info "분산 방식: ${DISTRIBUTE_METHOD}"
    echo ""
    
    # 출력 디렉토리
    mkdir -p "${LABEL_DIR}"
    mkdir -p "${VIZ_DIR}"
    
    # 이미지 분산 배치
    print_header "이미지 분산 배치"
    distribute_images "${IMAGE_DIR}" ${num_gpus} "${WORK_DIR}" "${DISTRIBUTE_METHOD}"
    
    # GPU 프로세스 실행
    echo ""
    print_header "GPU 프로세스 실행"
    
    for ((i=0; i<num_gpus; i++)); do
        local gpu_id=${GPU_LIST[$i]}
        local gpu_image_dir="${WORK_DIR}/gpu_${gpu_id}"
        
        run_gpu_process ${gpu_id} "${gpu_image_dir}"
        sleep 2
    done
    
    echo ""
    print_success "모든 프로세스 시작!"
    print_info "로그: ${LOG_DIR}"
    
    # 모니터링
    monitor_processes ${num_gpus}
    
    # 결과
    print_summary
    
    print_header "완료!"
    print_success "라벨: ${LABEL_DIR}"
    print_success "시각화: ${VIZ_DIR}"
    
    # 작업 디렉토리 정리 안내
    echo ""
    print_info "작업 디렉토리: ${WORK_DIR}"
    print_warning "완료 후 'rm -rf ${WORK_DIR}'로 정리하세요"
}

main "$@"