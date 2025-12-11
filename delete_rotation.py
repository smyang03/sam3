"""
EXIF Orientation 제거 도구
JPG 이미지 폴더의 모든 이미지에서 EXIF 회전 정보를 제거합니다.
"""

import os
from pathlib import Path
from PIL import Image
import piexif

def remove_exif_orientation(image_path, output_path=None, backup=True):
    """
    단일 이미지에서 EXIF Orientation 제거
    
    Args:
        image_path: 입력 이미지 경로
        output_path: 출력 이미지 경로 (Non e이면 덮어쓰기)
        backup: 원본 백업 여부
        
    Returns:
        bool: 성공 여부
    """
    try:
        # 이미지 열기
        img = Image.open(image_path)
        
        # EXIF 데이터 확인
        if 'exif' in img.info:
            exif_dict = piexif.load(img.info['exif'])
            
            # Orientation 태그 확인
            orientation = exif_dict.get('0th', {}).get(piexif.ImageIFD.Orientation)
            
            if orientation:
                print(f"  발견: Orientation = {orientation}")
                
                # Orientation 제거
                if piexif.ImageIFD.Orientation in exif_dict['0th']:
                    del exif_dict['0th'][piexif.ImageIFD.Orientation]
                
                # 수정된 EXIF를 bytes로 변환
                exif_bytes = piexif.dump(exif_dict)
            else:
                print(f"  Orientation 없음 (그대로 유지)")
                exif_bytes = img.info.get('exif')
        else:
            print(f"  EXIF 없음")
            exif_bytes = None
        
        # 출력 경로 설정
        if output_path is None:
            output_path = image_path
            
            # 백업 생성
            if backup and orientation:
                backup_path = str(image_path).rsplit('.', 1)[0] + '_backup.jpg'
                img.save(backup_path, 'JPEG', quality=95, exif=img.info.get('exif'))
                print(f"  백업 저장: {Path(backup_path).name}")
        
        # 이미지 저장 (EXIF Orientation 제거됨)
        if exif_bytes:
            img.save(output_path, 'JPEG', quality=95, exif=exif_bytes)
        else:
            img.save(output_path, 'JPEG', quality=95)
        
        return True
        
    except Exception as e:
        print(f"  오류: {e}")
        return False


def remove_exif_orientation_from_folder(
    input_folder, 
    output_folder=None, 
    backup=True,
    recursive=False
):
    """
    폴더 내 모든 JPG 이미지에서 EXIF Orientation 제거
    
    Args:
        input_folder: 입력 폴더 경로
        output_folder: 출력 폴더 경로 (None이면 덮어쓰기)
        backup: 원본 백업 여부 (덮어쓰기 모드일 때만)
        recursive: 하위 폴더 포함 여부
        
    Returns:
        dict: 처리 결과 통계
    """
    input_path = Path(input_folder)
    
    if not input_path.exists():
        print(f"❌ 폴더를 찾을 수 없습니다: {input_folder}")
        return None
    
    print("=" * 70)
    print("EXIF Orientation 제거 도구")
    print("=" * 70)
    print(f"입력 폴더: {input_folder}")
    
    if output_folder:
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)
        print(f"출력 폴더: {output_folder}")
        mode = "복사 모드"
    else:
        output_path = None
        print(f"출력 폴더: (덮어쓰기)")
        mode = "덮어쓰기 모드"
        if backup:
            print(f"백업: 활성화 (*_backup.jpg)")
    
    print(f"모드: {mode}")
    print(f"재귀 탐색: {'예' if recursive else '아니오'}")
    print("=" * 70)
    print()
    
    # 이미지 파일 수집
    if recursive:
        image_files = list(input_path.rglob("*.jpg")) + \
                     list(input_path.rglob("*.JPG")) + \
                     list(input_path.rglob("*.jpeg")) + \
                     list(input_path.rglob("*.JPEG"))
    else:
        image_files = list(input_path.glob("*.jpg")) + \
                     list(input_path.glob("*.JPG")) + \
                     list(input_path.glob("*.jpeg")) + \
                     list(input_path.glob("*.JPEG"))
    
    if len(image_files) == 0:
        print("❌ JPG 이미지를 찾을 수 없습니다.")
        return None
    
    print(f"📁 발견된 이미지: {len(image_files)}개\n")
    
    # 통계
    stats = {
        'total': len(image_files),
        'processed': 0,
        'removed': 0,
        'skipped': 0,
        'failed': 0
    }
    
    # 각 이미지 처리
    for idx, image_file in enumerate(image_files, 1):
        print(f"[{idx}/{len(image_files)}] {image_file.name}")
        
        # 출력 경로 결정
        if output_path:
            # 상대 경로 유지
            rel_path = image_file.relative_to(input_path)
            out_file = output_path / rel_path
            out_file.parent.mkdir(parents=True, exist_ok=True)
        else:
            out_file = None
        
        # EXIF 확인
        try:
            img = Image.open(image_file)
            has_orientation = False
            
            if 'exif' in img.info:
                exif_dict = piexif.load(img.info['exif'])
                orientation = exif_dict.get('0th', {}).get(piexif.ImageIFD.Orientation)
                if orientation:
                    has_orientation = True
                    stats['removed'] += 1
            
            if not has_orientation:
                stats['skipped'] += 1
                
        except Exception as e:
            print(f"  EXIF 확인 실패: {e}")
        
        # 처리
        success = remove_exif_orientation(
            image_file, 
            out_file, 
            backup=(backup and output_path is None)
        )
        
        if success:
            stats['processed'] += 1
        else:
            stats['failed'] += 1
        
        print()
    
    # 결과 출력
    print("=" * 70)
    print("처리 완료!")
    print("=" * 70)
    print(f"총 이미지:        {stats['total']}개")
    print(f"처리 완료:        {stats['processed']}개")
    print(f"Orientation 제거: {stats['removed']}개")
    print(f"변경 없음:        {stats['skipped']}개")
    if stats['failed'] > 0:
        print(f"실패:             {stats['failed']}개")
    print("=" * 70)
    
    return stats


def main():
    """메인 실행"""
    
    # ========== 설정 ==========
    CONFIG = {
        # 입력 폴더 (EXIF를 제거할 이미지들이 있는 폴더)
        "input_folder": "X:/박창현/pipe_lower_part/data/JPEGImages",
        
        # 출력 폴더 (None이면 원본 덮어쓰기)
        "output_folder": None,  # 예: "X:/박창현/pipe_lower_part/data/JPEGImages_no_exif"
        
        # 덮어쓰기 모드일 때 백업 생성 여부
        "backup": True,
        
        # 하위 폴더 포함 여부
        "recursive": False,
    }
    # ==========================
    
    print("\n⚠️  주의사항:")
    if CONFIG["output_folder"] is None:
        print("  - 원본 이미지를 덮어씁니다!")
        if CONFIG["backup"]:
            print("  - 백업 파일(*_backup.jpg)이 생성됩니다.")
        else:
            print("  - 백업이 생성되지 않습니다! (복구 불가)")
    else:
        print(f"  - 새 폴더에 복사본을 생성합니다: {CONFIG['output_folder']}")
        print("  - 원본은 그대로 유지됩니다.")
    
    # 확인
    response = input("\n계속하시겠습니까? (y/n): ")
    if response.lower() != 'y':
        print("취소되었습니다.")
        return
    
    print()
    
    # 실행
    result = remove_exif_orientation_from_folder(
        input_folder=CONFIG["input_folder"],
        output_folder=CONFIG["output_folder"],
        backup=CONFIG["backup"],
        recursive=CONFIG["recursive"]
    )
    
    if result:
        print("\n✅ 모든 작업이 완료되었습니다!")


if __name__ == "__main__":
    main()