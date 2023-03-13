#include "test_helper.hpp"

/* 4 - Test complex acos - double (Failed) */
/* 5 - Test complex acos - float (Failed) */

auto print_queue_info(sycl::queue &queue) -> void {
    std::cout << "backend: " << queue.get_backend() << "\n";
    std::cout << "platform: " << queue.get_device().get_platform().get_info<sycl::info::platform::name>() << "\n";
    std::cout << "device: " << queue.get_device().get_info<sycl::info::device::name>() << "\n";

    std::cout << std::endl;
}

TEMPLATE_TEST_CASE("Test complex acos", "[acos]", double, float, sycl::half) {

  using T = TestType;
  using std::make_tuple;

  sycl::queue Q;


  cmplx<T> input;
  bool is_error_checking;

  std::tie(input, is_error_checking) = GENERATE(table<cmplx<T>, bool>(
      {make_tuple(cmplx<T>{4.42, 2.02}, false),
       make_tuple(cmplx<T>{inf_val<T>, 2.02}, true),
       make_tuple(cmplx<T>{4.42, inf_val<T>}, true),
       make_tuple(cmplx<T>{inf_val<T>, inf_val<T>}, true),
       make_tuple(cmplx<T>{nan_val<T>, 2.02}, true),
       make_tuple(cmplx<T>{4.42, nan_val<T>}, true),
       make_tuple(cmplx<T>{nan_val<T>, nan_val<T>}, true),
       make_tuple(cmplx<T>{nan_val<T>, inf_val<T>}, true),
       make_tuple(cmplx<T>{inf_val<T>, nan_val<T>}, true)}));

  printf("\n");
  print_queue_info(Q);
  printf("%s : %s\n", __PRETTY_FUNCTION__, is_error_checking ? "true" : "false");

  auto std_in = init_std_complex(input);
  sycl::ext::cplx::complex<T> cplx_input{input.re, input.im};

  std::complex<T> std_out{input.re, input.im};
  auto *cplx_out = sycl::malloc_shared<sycl::ext::cplx::complex<T>>(1, Q);

  // Get std::complex output
  if (is_error_checking)
    std_out = std::acos(std_in);

  // Check cplx::complex output from device
  if (is_type_supported<T>(Q)) {
    if (is_error_checking) {
      Q.single_task(
          [=]() { cplx_out[0] = sycl::ext::cplx::acos<T>(cplx_input); });
    } else {
      Q.single_task([=]() {
        cplx_out[0] =
            sycl::ext::cplx::cos<T>(sycl::ext::cplx::acos<T>(cplx_input));
      });
    }
    Q.wait();

    printf("[+] CHECK RESULTS DEVICE\n");
    printf("\n");

    std::cout << "  sycl::cplx = (" << cplx_out->real() << ", " << cplx_out->imag() << ")\n";
    std::cout << "  std::cplx = (" << std_out.real() << ", " << std_out.imag() << ")\n";
    std::cout << "\n";

    check_results(cplx_out[0], std_out, /*tol_multiplier*/ 2);
  }

  // Check cplx::complex output from host
  if (is_error_checking)
    cplx_out[0] = sycl::ext::cplx::acos<T>(cplx_input);
  else
    cplx_out[0] = sycl::ext::cplx::cos<T>(sycl::ext::cplx::acos<T>(cplx_input));

  printf("[+] CHECK RESULTS HOST\n");
  printf("\n");

  std::cout << "  sycl::cplx = (" << cplx_out->real() << ", " << cplx_out->imag() << ")\n";
  std::cout << "  std::cplx = (" << std_out.real() << ", " << std_out.imag() << ")\n";
  std::cout << "\n";

  check_results(cplx_out[0], std_out, /*tol_multiplier*/ 2);

  sycl::free(cplx_out, Q);

  printf("--------------------------------------------------------------------------------");
  printf("\n");
}
