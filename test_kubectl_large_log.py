#!/usr/bin/env python3
"""Test script to reproduce kubectl exec large log streaming issue."""

import subprocess
import sys
import time

# Configuration - adjust these based on your setup
POD_NAME = "t-long-setup-run-by-56-9640c808-head"
NAMESPACE = "test-namespace"
CONTEXT = "kind-skypilot"

def test_1_direct_kubectl_no_pipe():
    """Test 1: Run kubectl directly with no Python subprocess.PIPE"""
    print("=" * 80)
    print("TEST 1: Direct kubectl (no PIPE) - simulating 200k lines")
    print("=" * 80)
    
    # Create a command that generates ~200k lines in the pod
    cmd = [
        "kubectl", "exec", "-i",
        "--pod-running-timeout", "30s",
        "-n", NAMESPACE,
        "--context", CONTEXT,
        f"pod/{POD_NAME}",
        "--",
        "/bin/bash", "-c",
        "for i in {1..200000}; do echo $i; done"
    ]
    
    print(f"Running: {' '.join(cmd)}")
    print(f"Started at: {time.time()}")
    start = time.time()
    
    # No PIPE - let it stream to terminal
    result = subprocess.run(cmd, check=False)
    
    elapsed = time.time() - start
    print(f"\nFinished at: {time.time()}")
    print(f"Elapsed: {elapsed:.2f}s")
    print(f"Return code: {result.returncode}")
    return result.returncode == 0


def test_2_kubectl_with_pipe_no_read():
    """Test 2: kubectl with subprocess.PIPE but NOT reading from it"""
    print("\n" + "=" * 80)
    print("TEST 2: kubectl with PIPE but NOT reading (should fail)")
    print("=" * 80)
    
    cmd = [
        "kubectl", "exec", "-i",
        "--pod-running-timeout", "30s",
        "-n", NAMESPACE,
        "--context", CONTEXT,
        f"pod/{POD_NAME}",
        "--",
        "/bin/bash", "-c",
        "for i in {1..200000}; do echo $i; done"
    ]
    
    print(f"Running: {' '.join(cmd)}")
    start = time.time()
    
    # Create PIPE but DON'T read from it - this should deadlock/fail
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL
    )
    
    # Just wait without reading
    returncode = proc.wait()
    
    elapsed = time.time() - start
    print(f"\nElapsed: {elapsed:.2f}s")
    print(f"Return code: {returncode}")
    return returncode == 0


def test_3_kubectl_with_pipe_and_read():
    """Test 3: kubectl with subprocess.PIPE and actively reading"""
    print("\n" + "=" * 80)
    print("TEST 3: kubectl with PIPE and actively reading (should work)")
    print("=" * 80)
    
    cmd = [
        "kubectl", "exec", "-i",
        "--pod-running-timeout", "30s",
        "-n", NAMESPACE,
        "--context", CONTEXT,
        f"pod/{POD_NAME}",
        "--",
        "/bin/bash", "-c",
        "for i in {1..200000}; do echo $i; done"
    ]
    
    print(f"Running: {' '.join(cmd)}")
    start = time.time()
    
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        text=True
    )
    
    # Actively read the output
    line_count = 0
    for line in proc.stdout:
        line_count += 1
        if line_count % 50000 == 0:
            print(f"  Read {line_count} lines so far...")
    
    returncode = proc.wait()
    
    elapsed = time.time() - start
    print(f"\nTotal lines read: {line_count}")
    print(f"Elapsed: {elapsed:.2f}s")
    print(f"Return code: {returncode}")
    return returncode == 0


def test_4_kubectl_with_shell_tee():
    """Test 4: kubectl with shell tee (mimics process_stream=False)"""
    print("\n" + "=" * 80)
    print("TEST 4: kubectl with shell | tee /dev/null (mimics process_stream=False)")
    print("=" * 80)
    
    # This mimics what happens when process_stream=False
    cmd = (
        f"kubectl exec -i --pod-running-timeout 30s "
        f"-n {NAMESPACE} --context {CONTEXT} pod/{POD_NAME} -- "
        f"/bin/bash -c 'for i in {{1..200000}}; do echo $i; done' "
        f"| tee /dev/null ; exit ${{PIPESTATUS[0]}}"
    )
    
    print(f"Running: {cmd}")
    start = time.time()
    
    # No PIPE, using shell redirection
    proc = subprocess.Popen(
        cmd,
        shell=True,
        executable='/bin/bash',
        stdin=subprocess.DEVNULL
    )
    
    returncode = proc.wait()
    
    elapsed = time.time() - start
    print(f"\nElapsed: {elapsed:.2f}s")
    print(f"Return code: {returncode}")
    return returncode == 0


def test_5_kubectl_with_context_simulation():
    """Test 5: kubectl with PIPE simulating context-aware streaming"""
    print("\n" + "=" * 80)
    print("TEST 5: kubectl with PIPE simulating context output (file write)")
    print("=" * 80)
    
    cmd = [
        "kubectl", "exec", "-i",
        "--pod-running-timeout", "30s",
        "-n", NAMESPACE,
        "--context", CONTEXT,
        f"pod/{POD_NAME}",
        "--",
        "/bin/bash", "-c",
        "for i in {1..200000}; do echo $i; done"
    ]
    
    print(f"Running: {' '.join(cmd)}")
    start = time.time()
    
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        text=True
    )
    
    # Write to a file like the real code does
    line_count = 0
    with open('/tmp/test_kubectl_output.log', 'w') as f:
        for line in proc.stdout:
            f.write(line)
            f.flush()  # Like the real code
            line_count += 1
            if line_count % 50000 == 0:
                print(f"  Wrote {line_count} lines so far...")
    
    returncode = proc.wait()
    
    elapsed = time.time() - start
    print(f"\nTotal lines written: {line_count}")
    print(f"Elapsed: {elapsed:.2f}s")
    print(f"Return code: {returncode}")
    return returncode == 0


def main():
    print(f"Testing kubectl exec with large log output")
    print(f"Pod: {POD_NAME}")
    print(f"Namespace: {NAMESPACE}")
    print(f"Context: {CONTEXT}")
    print()
    
    results = {}
    
    # Run tests in order
    tests = [
        ("Test 1: Direct kubectl (no PIPE)", test_1_direct_kubectl_no_pipe),
        ("Test 2: PIPE but no read (should fail)", test_2_kubectl_with_pipe_no_read),
        ("Test 3: PIPE with active reading", test_3_kubectl_with_pipe_and_read),
        ("Test 4: Shell tee (process_stream=False)", test_4_kubectl_with_shell_tee),
        ("Test 5: PIPE with file writing", test_5_kubectl_with_context_simulation),
    ]
    
    for name, test_func in tests:
        try:
            success = test_func()
            results[name] = "PASS" if success else "FAIL"
        except Exception as e:
            results[name] = f"ERROR: {e}"
            print(f"ERROR in {name}: {e}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for name, result in results.items():
        print(f"{name}: {result}")
    
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    print("If Test 2 FAILS but Test 3 PASSES:")
    print("  => The issue is NOT reading from subprocess.PIPE causes deadlock")
    print("  => Fix: Always set process_stream=True or always read from PIPE")
    print()
    print("If Test 4 FAILS:")
    print("  => Shell tee redirection doesn't help with large output")
    print("  => The issue is with kubectl exec itself or shell buffering")
    print()
    print("If Test 5 FAILS:")
    print("  => Writing to file while reading from PIPE causes issues")
    print("  => May need async I/O or separate thread for file writing")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ["--pod", "--help"]:
        print("Usage: python test_kubectl_large_log.py")
        print("       Edit the script to set POD_NAME, NAMESPACE, CONTEXT")
        sys.exit(0)
    
    main()

