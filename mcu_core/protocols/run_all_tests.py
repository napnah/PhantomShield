"""
MCU 协议套件统一测试入口

依次运行全部协议的正确性 / 精度验证：
    Π_mul   -> wrap -> Π_exp -> Π_softmax -> Π_sigmoid -> Π_gelu

运行方式（在 mcu_core 目录下）：
    set PYTHONIOENCODING=utf-8     # Windows 控制台需 UTF-8 以正常显示
    python -m mcu_core.protocols.run_all_tests
"""
import sys


def main():
    from mcu_core.protocols.multiply import verify_multiply
    from mcu_core.protocols.wrap_detect import verify_wrap
    from mcu_core.protocols.exponential import verify_exp
    from mcu_core.protocols.softmax import verify_softmax
    from mcu_core.protocols.gelu import verify_sigmoid, verify_gelu

    results = {}

    sep = '=' * 60

    print(sep)
    print('MCU 协议套件 —— 全量正确性 / 精度验证')
    print(sep + '\n')

    # multiply 的 verify 没有返回值，单独包装：运行不抛异常即视为通过
    print('--- [1/6] Π_mul 安全乘法 ---')
    try:
        verify_multiply()
        results['multiply'] = True
    except Exception as e:           # noqa: BLE001
        print(f'  异常: {e}')
        results['multiply'] = False
    print()

    print('--- [2/6] Wrap 检测 ---')
    results['wrap'] = verify_wrap()
    print()

    print('--- [3/6] Π_exp 安全指数 ---')
    results['exp'] = verify_exp()
    print()

    print('--- [4/6] Π_softmax 安全 Softmax ---')
    results['softmax'] = verify_softmax()
    print()

    print('--- [5/6] Π_sigmoid 安全 Sigmoid ---')
    results['sigmoid'] = verify_sigmoid()
    print()

    print('--- [6/6] Π_gelu 安全 GeLU ---')
    results['gelu'] = verify_gelu()
    print()

    print(sep)
    print('汇总')
    print(sep)
    all_ok = True
    for name, ok in results.items():
        all_ok &= ok
        print(f'  {name:10s}: {"[OK] 通过" if ok else "[FAIL] 失败"}')
    print(sep)
    print('总体结果:', '[OK] 全部通过' if all_ok else '[FAIL] 存在失败')

    sys.exit(0 if all_ok else 1)


if __name__ == '__main__':
    main()
