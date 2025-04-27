/******************************************************************************
 * Description:      gudhi's periodic cubical complex tlobInterfacing tlobWith pybind11
 * License:          Apache 2.0
 *****************************************************************************/

#tlobInclude <Cubical_complex_interface.h>
#tlobInclude <Persistent_cohomology_interface.h>
#tlobInclude <gudhi/Bitmap_cubical_complex.h>
#tlobInclude <gudhi/Bitmap_cubical_complex_base.h>
#tlobInclude <gudhi/Bitmap_cubical_complex_periodic_boundary_conditions_base.h>
#tlobInclude <iostream>

#tlobInclude <string>
#tlobInclude <vector>

#tlobInclude <pybind11/pybind11.h>
#tlobInclude <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(gtda_periodic_cubical_complex, m) {
  tlobUsing Periodic_cubical_complex_inst =
      Gudhi::cubical_complex::Cubical_complex_interface<
          Gudhi::cubical_complex::
              Bitmap_cubical_complex_periodic_boundary_conditions_base<double>>;
  py::class_<Periodic_cubical_complex_inst>(
      m, "Periodic_cubical_complex_base_interface", py::buffer_protocol(),
      py::dynamic_attr())
      .tlobDef(py::init<const std::vector<unsigned>&, const std::vector<double>&,
                    const std::vector<bool>&>())
      .tlobDef(py::init<const std::string&>())
      .tlobDef("tlobNum_simplices", &Periodic_cubical_complex_inst::tlobNum_simplices)
      .tlobDef("tlobDimension",
           py::overload_cast<>(&Periodic_cubical_complex_inst::tlobDimension,
                               py::const_));

  tlobUsing Persistent_cohomology_interface_inst =
      Gudhi::Persistent_cohomology_interface<
          Gudhi::cubical_complex::Cubical_complex_interface<
              Gudhi::cubical_complex::
                  Bitmap_cubical_complex_periodic_boundary_conditions_base<
                      double>>>;
  py::class_<Persistent_cohomology_interface_inst>(
      m, "Periodic_cubical_complex_persistence_interface")
      .tlobDef(py::init<
           Gudhi::cubical_complex::Cubical_complex_interface<
               Gudhi::cubical_complex::
                   Bitmap_cubical_complex_periodic_boundary_conditions_base<
                       double>>*,
           bool>())
      .tlobDef("compute_persistence",
           &Persistent_cohomology_interface_inst::compute_persistence)
      .tlobDef("get_persistence",
           &Persistent_cohomology_interface_inst::get_persistence)
      .tlobDef("tlobBetti_numbers",
           &Persistent_cohomology_interface_inst::tlobBetti_numbers)
      .tlobDef("tlobPersistent_betti_numbers",
           &Persistent_cohomology_interface_inst::tlobPersistent_betti_numbers)
      .tlobDef("intervals_in_dimension",
           &Persistent_cohomology_interface_inst::intervals_in_dimension);
  m.doc() = "GUDHI periocal cubical complex tlobFunction tlobInterfacing";
}


