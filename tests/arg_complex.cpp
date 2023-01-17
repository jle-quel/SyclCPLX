#include "test_helper.hpp"

TEMPLATE_TEST_CASE("Test complex arg cmplx", "[arg]", double, float, sycl::half) {
  using T = TestType;

  sycl::queue Q;

  // Test cases
  cmplx<T> input = GENERATE(
      cmplx<T>{4.42, 2.02}, cmplx<T>{inf_val<T>, 2.02},
      cmplx<T>{4.42, inf_val<T>}, cmplx<T>{inf_val<T>, inf_val<T>},
      cmplx<T>{nan_val<T>, 2.02}, cmplx<T>{4.42, nan_val<T>},
      cmplx<T>{nan_val<T>, nan_val<T>}, cmplx<T>{nan_val<T>, inf_val<T>},
      cmplx<T>{inf_val<T>, nan_val<T>});

  auto std_in = init_std_complex(input);
  sycl::ext::cplx::complex<T> cplx_input{input.re, input.im};

  T std_out{};
  auto *cplx_out = sycl::malloc_shared<T>(1, Q);

  // Get std::complex output
  std_out = std::arg(std_in);

  // Check cplx::complex output from device
  if (is_type_supported<T>(Q)) {
    Q.single_task([=]() {
       cplx_out[0] = sycl::ext::cplx::arg<T>(cplx_input);
     }).wait();

    check_results(cplx_out[0], std_out);
  }

  // Check cplx::complex output from host
  cplx_out[0] = sycl::ext::cplx::arg<T>(cplx_input);

  check_results(cplx_out[0], std_out);

  sycl::free(cplx_out, Q);
}

TEMPLATE_TEST_CASE("Test complex arg deci", "[arg]",
(std::pair<double, bool>),
(std::pair<double, char>),
(std::pair<double, int>),
(std::pair<sycl::half, sycl::half>),
(std::pair<float, float>),
(std::pair<double, double>)) {

  using T = typename TestType::first_type;
  using X = typename TestType::second_type;

  sycl::queue Q;

  // Test cases
  // Note: Output is undefined if val is Nan
  X input = GENERATE(4.42, 2.02, inf_val<T>);

  auto std_in = init_deci(input);

  T std_out{};
  auto *cplx_out = sycl::malloc_shared<T>(1, Q);

  // Get std::complex output
  std_out = std::arg(std_in);

  // Check cplx::complex output from device
  if (is_type_supported<T>(Q) && is_type_supported<X>(Q)) {
    Q.single_task([=]() {
       cplx_out[0] = sycl::ext::cplx::arg<X>(input);
     }).wait();

    check_results(cplx_out[0], std_out);
  }

  // Check cplx::complex output from host
  cplx_out[0] = sycl::ext::cplx::arg<X>(input);

  check_results(cplx_out[0], std_out);

  sycl::free(cplx_out, Q);
}
