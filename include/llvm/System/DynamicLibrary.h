//===-- llvm/System/DynamicLibrary.h - Portable Dynamic Library -*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Reid Spencer and is distributed under the 
// University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file declares the sys::DynamicLibrary class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SYSTEM_DYNAMIC_LIBRARY_H
#define LLVM_SYSTEM_DYNAMIC_LIBRARY_H

#include "llvm/System/Path.h"
#include <string>

namespace llvm {
namespace sys {

  /// This class provides a portable interface to dynamic libraries which also
  /// might be known as shared libraries, shared objects, dynamic shared 
  /// objects, or dynamic link libraries. Regardless of the terminology or the
  /// operating system interface, this class provides a portable interface that
  /// allows dynamic libraries to be loaded and and searched for externally 
  /// defined symbols. This is typically used to provide "plug-in" support.
  /// @since 1.4
  /// @brief Portable dynamic library abstraction.
  class DynamicLibrary {
    /// @name Constructors
    /// @{
    public:
      /// Construct a DynamicLibrary that represents the currently executing
      /// program. The program must have been linked with -export-dynamic or
      /// -dlopen self for this to work. Any symbols retrieved with the 
      /// GetAddressOfSymbol function will refer to the program not to any
      /// library.
      /// @throws std::string indicating why the program couldn't be opened.
      /// @brief Open program as dynamic library.
      DynamicLibrary();

      /// This is the constructor for DynamicLibrary instances. It will open
      /// the dynamic library specified by the \filename Path.
      /// @throws std::string indicating why the library couldn't be opened.
      /// @brief Open a dynamic library.
      DynamicLibrary(const char* filename);

      /// After destruction, the symbols of the library will no longer be
      /// available to the program. It is important to make sure the lifespan
      /// of a DynamicLibrary exceeds the lifetime of the pointers returned 
      /// by the GetAddressOfSymbol otherwise the program may walk off into 
      /// uncharted territory.
      /// @see GetAddressOfSymbol.
      /// @brief Closes the DynamicLibrary
      ~DynamicLibrary();

    /// @}
    /// @name Accessors
    /// @{
    public:
      /// Looks up a \p symbolName in the DynamicLibrary and returns its address
      /// if it exists. If the symbol does not exist, returns (void*)0.
      /// @returns the address of the symbol or 0.
      /// @brief Get the address of a symbol in the DynamicLibrary.
      void* GetAddressOfSymbol(const char* symbolName);

      /// @brief Convenience function for C++ophiles.
      void* GetAddressOfSymbol(const std::string& symbolName) {
        return GetAddressOfSymbol(symbolName.c_str());
      }

    /// @}
    /// @name Implementation
    /// @{
    protected:
      void* handle;  // Opaque handle for information about the library

      DynamicLibrary(const DynamicLibrary&); ///< Do not implement
      DynamicLibrary& operator=(const DynamicLibrary&); ///< Do not implement
    /// @}
  };

} // End sys namespace
} // End llvm namespace

#endif // LLVM_SYSTEM_DYNAMIC_LIBRARY_H
