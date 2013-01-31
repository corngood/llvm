; RUN: llc -verify-machineinstrs < %s -march=aarch64 -fp-contract=fast | FileCheck %s

declare float @llvm.fma.f32(float, float, float)
declare double @llvm.fma.f64(double, double, double)

define float @test_fmadd(float %a, float %b, float %c) {
; CHECK: test_fmadd:
  %val = call float @llvm.fma.f32(float %a, float %b, float %c)
; CHECK: fmadd {{s[0-9]+}}, {{s[0-9]+}}, {{s[0-9]+}}, {{s[0-9]+}}
  ret float %val
}

define float @test_fmsub(float %a, float %b, float %c) {
; CHECK: test_fmsub:
  %nega = fsub float -0.0, %a
  %val = call float @llvm.fma.f32(float %nega, float %b, float %c)
; CHECK: fmsub {{s[0-9]+}}, {{s[0-9]+}}, {{s[0-9]+}}, {{s[0-9]+}}
  ret float %val
}

define float @test_fnmadd(float %a, float %b, float %c) {
; CHECK: test_fnmadd:
  %negc = fsub float -0.0, %c
  %val = call float @llvm.fma.f32(float %a, float %b, float %negc)
; CHECK: fnmadd {{s[0-9]+}}, {{s[0-9]+}}, {{s[0-9]+}}, {{s[0-9]+}}
  ret float %val
}

define float @test_fnmsub(float %a, float %b, float %c) {
; CHECK: test_fnmsub:
  %nega = fsub float -0.0, %a
  %negc = fsub float -0.0, %c
  %val = call float @llvm.fma.f32(float %nega, float %b, float %negc)
; CHECK: fnmsub {{s[0-9]+}}, {{s[0-9]+}}, {{s[0-9]+}}, {{s[0-9]+}}
  ret float %val
}

define double @testd_fmadd(double %a, double %b, double %c) {
; CHECK: testd_fmadd:
  %val = call double @llvm.fma.f64(double %a, double %b, double %c)
; CHECK: fmadd {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}
  ret double %val
}

define double @testd_fmsub(double %a, double %b, double %c) {
; CHECK: testd_fmsub:
  %nega = fsub double -0.0, %a
  %val = call double @llvm.fma.f64(double %nega, double %b, double %c)
; CHECK: fmsub {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}
  ret double %val
}

define double @testd_fnmadd(double %a, double %b, double %c) {
; CHECK: testd_fnmadd:
  %negc = fsub double -0.0, %c
  %val = call double @llvm.fma.f64(double %a, double %b, double %negc)
; CHECK: fnmadd {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}
  ret double %val
}

define double @testd_fnmsub(double %a, double %b, double %c) {
; CHECK: testd_fnmsub:
  %nega = fsub double -0.0, %a
  %negc = fsub double -0.0, %c
  %val = call double @llvm.fma.f64(double %nega, double %b, double %negc)
; CHECK: fnmsub {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}
  ret double %val
}

define float @test_fmadd_unfused(float %a, float %b, float %c) {
; CHECK: test_fmadd_unfused:
  %prod = fmul float %b, %c
  %sum = fadd float %a, %prod
; CHECK: fmadd {{s[0-9]+}}, {{s[0-9]+}}, {{s[0-9]+}}, {{s[0-9]+}}
  ret float %sum
}

define float @test_fmsub_unfused(float %a, float %b, float %c) {
; CHECK: test_fmsub_unfused:
  %prod = fmul float %b, %c
  %diff = fsub float %a, %prod
; CHECK: fmsub {{s[0-9]+}}, {{s[0-9]+}}, {{s[0-9]+}}, {{s[0-9]+}}
  ret float %diff
}

define float @test_fnmadd_unfused(float %a, float %b, float %c) {
; CHECK: test_fnmadd_unfused:
  %nega = fsub float -0.0, %a
  %prod = fmul float %b, %c
  %sum = fadd float %nega, %prod
; CHECK: fnmadd {{s[0-9]+}}, {{s[0-9]+}}, {{s[0-9]+}}, {{s[0-9]+}}
  ret float %sum
}

define float @test_fnmsub_unfused(float %a, float %b, float %c) {
; CHECK: test_fnmsub_unfused:
  %nega = fsub float -0.0, %a
  %prod = fmul float %b, %c
  %diff = fsub float %nega, %prod
; CHECK: fnmsub {{s[0-9]+}}, {{s[0-9]+}}, {{s[0-9]+}}, {{s[0-9]+}}
  ret float %diff
}
