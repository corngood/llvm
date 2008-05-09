//===-LTOModule.cpp - LLVM Link Time Optimizer ----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file implements the Link Time Optimization library. This library is 
// intended to be used by linker to optimize code at link time.
//
//===----------------------------------------------------------------------===//

#include "LTOModule.h"

#include "llvm/Module.h"
#include "llvm/ModuleProvider.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/Bitcode/ReaderWriter.h"
#include "llvm/Support/SystemUtils.h"
#include "llvm/Support/Mangler.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/System/Path.h"
#include "llvm/System/Process.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetMachineRegistry.h"
#include "llvm/Target/TargetAsmInfo.h"


#include <fstream>

using namespace llvm;

bool LTOModule::isBitcodeFile(const void* mem, size_t length)
{
    return ( llvm::sys::IdentifyFileType((char*)mem, length) 
                                            == llvm::sys::Bitcode_FileType );
}

bool LTOModule::isBitcodeFile(const char* path)
{
    return llvm::sys::Path(path).isBitcodeFile();
}

bool LTOModule::isBitcodeFileForTarget(const void* mem, size_t length,
                                       const char* triplePrefix) 
{
    MemoryBuffer* buffer = makeBuffer(mem, length);
    if ( buffer == NULL )
        return false;
    return isTargetMatch(buffer, triplePrefix);
}


bool LTOModule::isBitcodeFileForTarget(const char* path,
                                       const char* triplePrefix) 
{
    MemoryBuffer *buffer = MemoryBuffer::getFile(path);
    if (buffer == NULL)
        return false;
    return isTargetMatch(buffer, triplePrefix);
}

// takes ownership of buffer
bool LTOModule::isTargetMatch(MemoryBuffer* buffer, const char* triplePrefix)
{
    OwningPtr<ModuleProvider> mp(getBitcodeModuleProvider(buffer));
    // on success, mp owns buffer and both are deleted at end of this method
    if ( !mp ) {
        delete buffer;
        return false;
    }
    std::string actualTarget = mp->getModule()->getTargetTriple();
    return ( strncmp(actualTarget.c_str(), triplePrefix, 
                    strlen(triplePrefix)) == 0);
}


LTOModule::LTOModule(Module* m, TargetMachine* t) 
 : _module(m), _target(t), _symbolsParsed(false)
{
}

LTOModule* LTOModule::makeLTOModule(const char* path, std::string& errMsg)
{
    OwningPtr<MemoryBuffer> buffer(MemoryBuffer::getFile(path, &errMsg));
    if ( !buffer )
        return NULL;
    return makeLTOModule(buffer.get(), errMsg);
}

/// makeBuffer - create a MemoryBuffer from a memory range.
/// MemoryBuffer requires the byte past end of the buffer to be a zero.
/// We might get lucky and already be that way, otherwise make a copy.
/// Also if next byte is on a different page, don't assume it is readable.
MemoryBuffer* LTOModule::makeBuffer(const void* mem, size_t length)
{
	const char* startPtr = (char*)mem;
	const char* endPtr = startPtr+length;
	if ( (((uintptr_t)endPtr & (sys::Process::GetPageSize()-1)) == 0) 
	    || (*endPtr != 0) ) 
		return MemoryBuffer::getMemBufferCopy(startPtr, endPtr);
	else
		return MemoryBuffer::getMemBuffer(startPtr, endPtr);
}


LTOModule* LTOModule::makeLTOModule(const void* mem, size_t length, 
                                                        std::string& errMsg)
{
    OwningPtr<MemoryBuffer> buffer(makeBuffer(mem, length));
    if ( !buffer )
        return NULL;
    return makeLTOModule(buffer.get(), errMsg);
}

LTOModule* LTOModule::makeLTOModule(MemoryBuffer* buffer, std::string& errMsg)
{
    // parse bitcode buffer
    OwningPtr<Module> m(ParseBitcodeFile(buffer, &errMsg));
    if ( !m )
        return NULL;
    // find machine architecture for this module
    const TargetMachineRegistry::entry* march = 
            TargetMachineRegistry::getClosestStaticTargetForModule(*m, errMsg);
    if ( march == NULL ) 
        return NULL;
    // construct LTModule, hand over ownership of module and target
    std::string     features;
    TargetMachine*  target = march->CtorFn(*m, features);
    return new LTOModule(m.take(), target);
}


const char* LTOModule::getTargetTriple()
{
    return _module->getTargetTriple().c_str();
}

void LTOModule::addDefinedFunctionSymbol(Function* f, Mangler &mangler)
{
    // add to list of defined symbols
    addDefinedSymbol(f, mangler, true); 

    // add external symbols referenced by this function.
    for (Function::iterator b = f->begin(); b != f->end(); ++b) {
        for (BasicBlock::iterator i = b->begin(); i != b->end(); ++i) {
            for (unsigned count = 0, total = i->getNumOperands(); 
                                        count != total; ++count) {
                findExternalRefs(i->getOperand(count), mangler);
            }
        }
    }
}

void LTOModule::addDefinedDataSymbol(GlobalValue* v, Mangler &mangler)
{    
    // add to list of defined symbols
    addDefinedSymbol(v, mangler, false); 

    // add external symbols referenced by this data.
    for (unsigned count = 0, total = v->getNumOperands();\
                                                count != total; ++count) {
        findExternalRefs(v->getOperand(count), mangler);
    }
}


void LTOModule::addDefinedSymbol(GlobalValue* def, Mangler &mangler, 
                                bool isFunction)
{    
    // string is owned by _defines
    const char* symbolName = ::strdup(mangler.getValueName(def).c_str());
    
    // set alignment part log2() can have rounding errors
    uint32_t align = def->getAlignment();
    uint32_t attr = align ? CountTrailingZeros_32(def->getAlignment()) : 0;
    
    // set permissions part
    if ( isFunction )
        attr |= LTO_SYMBOL_PERMISSIONS_CODE;
    else {
        GlobalVariable* gv = dyn_cast<GlobalVariable>(def);
        if ( (gv != NULL) && gv->isConstant() )
            attr |= LTO_SYMBOL_PERMISSIONS_RODATA;
        else
            attr |= LTO_SYMBOL_PERMISSIONS_DATA;
    }
    
    // set definition part 
    if ( def->hasWeakLinkage() || def->hasLinkOnceLinkage() ) {
        // lvm bitcode does not differenciate between weak def data 
        // and tentative definitions!
        // HACK HACK HACK
        // C++ does not use tentative definitions, but does use weak symbols
        // so guess that anything that looks like a C++ symbol is weak and others
        // are tentative definitions
        if ( (strncmp(symbolName, "__Z", 3) == 0) )
            attr |= LTO_SYMBOL_DEFINITION_WEAK;
        else {
            attr |= LTO_SYMBOL_DEFINITION_TENTATIVE;
        }
    }
    else { 
        attr |= LTO_SYMBOL_DEFINITION_REGULAR;
    }
    
    // set scope part
    if ( def->hasHiddenVisibility() )
        attr |= LTO_SYMBOL_SCOPE_HIDDEN;
    else if ( def->hasExternalLinkage() || def->hasWeakLinkage() )
        attr |= LTO_SYMBOL_SCOPE_DEFAULT;
    else
        attr |= LTO_SYMBOL_SCOPE_INTERNAL;

    // add to table of symbols
    NameAndAttributes info;
    info.name = symbolName;
    info.attributes = (lto_symbol_attributes)attr;
    _symbols.push_back(info);
    _defines[info.name] = 1;
}


void LTOModule::addPotentialUndefinedSymbol(GlobalValue* decl, Mangler &mangler)
{   
   const char* name = mangler.getValueName(decl).c_str();
    // ignore all llvm.* symbols
    if ( strncmp(name, "llvm.", 5) != 0 ) {
        _undefines[name] = 1;
    }
}



// Find exeternal symbols referenced by VALUE. This is a recursive function.
void LTOModule::findExternalRefs(Value* value, Mangler &mangler) {

    if (GlobalValue* gv = dyn_cast<GlobalValue>(value)) {
        if ( !gv->hasExternalLinkage() )
            addPotentialUndefinedSymbol(gv, mangler);
    }

    // GlobalValue, even with InternalLinkage type, may have operands with 
    // ExternalLinkage type. Do not ignore these operands.
    if (Constant* c = dyn_cast<Constant>(value)) {
        // Handle ConstantExpr, ConstantStruct, ConstantArry etc..
        for (unsigned i = 0, e = c->getNumOperands(); i != e; ++i)
            findExternalRefs(c->getOperand(i), mangler);
    }
}

void LTOModule::lazyParseSymbols()
{
    if ( !_symbolsParsed ) {
        _symbolsParsed = true;
        
        // Use mangler to add GlobalPrefix to names to match linker names.
        Mangler mangler(*_module, _target->getTargetAsmInfo()->getGlobalPrefix());

        // add functions
        for (Module::iterator f = _module->begin(); f != _module->end(); ++f) {
            if ( f->isDeclaration() ) 
                addPotentialUndefinedSymbol(f, mangler);
            else 
                addDefinedFunctionSymbol(f, mangler);
        }
        
        // add data 
        for (Module::global_iterator v = _module->global_begin(), 
                                    e = _module->global_end(); v !=  e; ++v) {
            if ( v->isDeclaration() ) 
                addPotentialUndefinedSymbol(v, mangler);
            else 
                addDefinedDataSymbol(v, mangler);
        }

        // make symbols for all undefines
        for (StringSet::iterator it=_undefines.begin(); 
                                                it != _undefines.end(); ++it) {
            // if this symbol also has a definition, then don't make an undefine
            // because it is a tentative definition
            if ( _defines.count(it->getKeyData(), it->getKeyData()+
                                                  it->getKeyLength()) == 0 ) {
                NameAndAttributes info;
                info.name = it->getKeyData();
                info.attributes = LTO_SYMBOL_DEFINITION_UNDEFINED;
                _symbols.push_back(info);
            }
        }
    }    
}


uint32_t LTOModule::getSymbolCount()
{
    lazyParseSymbols();
    return _symbols.size();
}


lto_symbol_attributes LTOModule::getSymbolAttributes(uint32_t index)
{
    lazyParseSymbols();
    if ( index < _symbols.size() )
        return _symbols[index].attributes;
    else
        return lto_symbol_attributes(0);
}

const char* LTOModule::getSymbolName(uint32_t index)
{
    lazyParseSymbols();
    if ( index < _symbols.size() )
        return _symbols[index].name;
    else
        return NULL;
}

