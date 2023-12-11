/*
 * Copyright 2023 The DAPHNE Consortium
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cstddef>
#include <iostream>
#include <memory>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "compiler/utils/CompilerUtils.h"
#include "compiler/utils/LoweringUtils.h"
#include "ir/daphneir/Daphne.h"
#include "ir/daphneir/Passes.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_os_ostream.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/LinalgToStandard/LinalgToStandard.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/UseDefLists.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

static constexpr int ROW = 0;
static constexpr int COL = 1;

llvm::SmallVector<AffineForOp, 3> affineMatMul(mlir::Value &lhs, mlir::Value &rhs, mlir::Value &output,
                  ConversionPatternRewriter &rewriter, mlir::Location loc,
                  ArrayRef<int64_t> lhsShape, ArrayRef<int64_t> rhsShape,
                  mlir::MLIRContext *ctx) {
    llvm::SmallVector<AffineForOp, 3> loops;
    // row loop
    auto rowLoop = rewriter.create<AffineForOp>(loc, 0, lhsShape[ROW], 1);
    /* for (Operation &nested : *rowLoop.getBody()) {
        rewriter.eraseOp(&nested);
    } */
    // row loop body
    rewriter.setInsertionPointToStart(rowLoop.getBody());
    // fma loop
    auto fmaLoop = rewriter.create<AffineForOp>(loc, 0, rhsShape[ROW], 1);
    /* for (Operation &nested : *fmaLoop.getBody()) {
        rewriter.eraseOp(&nested);
    } */
    // inner loop body
    rewriter.setInsertionPointToStart(fmaLoop.getBody());
    // col loop
    auto colLoop = rewriter.create<AffineForOp>(loc, 0, rhsShape[COL], 1);
    /* for (Operation &nested : *colLoop.getBody()) {
        rewriter.eraseOp(&nested);
    } */
    // col loop body
    rewriter.setInsertionPointToStart(colLoop.getBody());

    // load
    mlir::Value a = rewriter.create<AffineLoadOp>(
        loc, lhs, ValueRange{rowLoop.getInductionVar(), fmaLoop.getInductionVar()});
    mlir::Value b = rewriter.create<AffineLoadOp>(
        loc, rhs, ValueRange{fmaLoop.getInductionVar(), colLoop.getInductionVar()});
    mlir::Value c = rewriter.create<AffineLoadOp>(
        loc, output, ValueRange{rowLoop.getInductionVar(), colLoop.getInductionVar()});

    // fma
    mlir::Value fma = rewriter.create<mlir::math::FmaOp>(loc, a, b, c);

    // store
    rewriter.create<AffineStoreOp>(loc, fma, output,
                                     ValueRange{rowLoop.getInductionVar(), colLoop.getInductionVar()});

    // AffineYieldOp at end of loop blocks
    //rewriter.create<AffineYieldOp>(loc);
    rewriter.setInsertionPointAfter(colLoop);
    //rewriter.create<AffineYieldOp>(loc);
    rewriter.setInsertionPointAfter(fmaLoop);
    //rewriter.create<AffineYieldOp>(loc);
    rewriter.setInsertionPointAfter(rowLoop);

    //colLoop.getLoopBody().back().erase();
    //rewriter.eraseOp(&*rowLoop.getBody()->end());

    loops.push_back(rowLoop);
    loops.push_back(fmaLoop);
    loops.push_back(colLoop);
    return loops;
}

void getPerfectlyNestedLoops2(SmallVectorImpl<AffineForOp> &nestedLoops,
                                   AffineForOp root) {
  for (unsigned i = 0; i < std::numeric_limits<unsigned>::max(); ++i) {
    nestedLoops.push_back(root);
    Block &body = root.getRegion().front();
    std::string s;
    llvm::raw_string_ostream os(s);
    body.print(os);
    std::cout << "Root name " << root->getName().getStringRef().str() << "; " << s << std::endl;
    root->dump();
    std::cout << std::endl;
    if (body.begin() != std::prev(body.end(), 2))
    {
        std::cout << "Children ";
        for (auto o=body.begin(); o!=body.end(); o++)
        {
            
            std::cout << o->getName().getStringRef().str() << " ";
        }
        std::cout << std::endl;
        return;
    }

    root = dyn_cast<AffineForOp>(&body.front());
    if (!root){
        std::cout << "B End at " << i << std::endl;
      return;}
  }
}

class MatMulLowering : public OpConversionPattern<daphne::MatMulOp> {
   public:
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(
        daphne::MatMulOp op, OpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override {
        auto loc = op->getLoc();
        mlir::daphne::MatrixType lhsMatrixType =
            adaptor.getLhs().getType().dyn_cast<mlir::daphne::MatrixType>();
        mlir::daphne::MatrixType rhsMatrixType =
            adaptor.getRhs().getType().dyn_cast<mlir::daphne::MatrixType>();

        auto lhsRows = lhsMatrixType.getNumRows();
        auto lhsCols = lhsMatrixType.getNumCols();

        auto rhsRows = rhsMatrixType.getNumRows();
        auto rhsCols = rhsMatrixType.getNumCols();

        auto matrixElementType = lhsMatrixType.getElementType();

        // TODO(phil): if shape is unknown, e.g., row/col = -1 we currently
        // can't create a MemRefType
        auto lhsMemRefType =
            mlir::MemRefType::get({lhsRows, lhsCols}, matrixElementType);
        auto rhsMemRefType =
            mlir::MemRefType::get({rhsRows, rhsCols}, matrixElementType);

        mlir::MemRefType outputMemRefType =
            mlir::MemRefType::get({lhsRows, rhsCols}, matrixElementType);

        // daphne::Matrix -> memref
        mlir::Value lhs =
            rewriter.create<mlir::daphne::ConvertDenseMatrixToMemRef>(
                op->getLoc(), lhsMemRefType, adaptor.getLhs());
        mlir::Value rhs =
            rewriter.create<mlir::daphne::ConvertDenseMatrixToMemRef>(
                op->getLoc(), rhsMemRefType, adaptor.getRhs());

        // Alloc output memref
        mlir::Value outputMemRef =
            insertMemRefAlloc(outputMemRefType, loc, rewriter);

        // Fill the output MemRef
        affineFillMemRef(0.0, rewriter, loc, outputMemRefType.getShape(),
                         op->getContext(), outputMemRef, matrixElementType);
        // Do the actual MatMul with hand built codegen
        auto loops = affineMatMul(lhs, rhs, outputMemRef, rewriter, loc,
                     lhsMemRefType.getShape(), rhsMemRefType.getShape(),
                     op->getContext());
        daphne::createPrintIRPass();
        llvm::SmallVector<AffineForOp> loopNest;
        getPerfectlyNestedLoops2(loopNest, loops.front());
        llvm::SmallVector<AffineForOp> loopNest3;
        getPerfectlyNestedLoops2(loopNest3, loops.back());
        llvm::SmallVector<AffineForOp> loopNest2;
        loops.pop_back();
        getPerfectlyNestedLoops2(loopNest2, loops.back());
        std::cout << "Outer loop: " << isPerfectlyNested(loopNest)  << loopNest.size() << std::endl;
        std::cout << "FMA loop: " << isPerfectlyNested(loopNest2)  << loopNest2.size() << std::endl;
        std::cout << "Inner Loop : " << isPerfectlyNested(loopNest3)  << loopNest3.size() << std::endl;
        
        /* if (failed(tilePerfectlyNested(loopNest, {4, 8}))){
            std::cout << "Tiling the loops failed" << std::endl;
        }; */

        mlir::Value DM = convertMemRefToDenseMatrix(loc, rewriter, outputMemRef,
                                                    op.getType());

        rewriter.replaceOp(op, DM);
        return success();
    }
};

namespace {
/**
 * @brief The MatMulLoweringPass rewrites the MatMulOp from the DaphneDialect
 * to a affine loop structure implementing a naive iterative matrix
 * multiplication.
 *
 * The naive iterative algorithm is simply a perfectly nested
 * loop algorithm running in O(n^3) performing the 3 load operations in it's
 * inner loop body, calculates an FMA and stores the result in the output
 * matrix.
 */
struct MatMulLoweringPass
    : public mlir::PassWrapper<MatMulLoweringPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
    explicit MatMulLoweringPass() {}

    StringRef getArgument() const final { return "lower-mm"; }
    StringRef getDescription() const final {
        return "This pass lowers the MatMulOp to an affine loop structure "
               "performing a naive iterative matrix multiplication.";
    }

    void getDependentDialects(mlir::DialectRegistry &registry) const override {
        registry.insert<mlir::LLVM::LLVMDialect, mlir::AffineDialect,
                        mlir::memref::MemRefDialect>();
    }
    void runOnOperation() final;
};
}  // end anonymous namespace

void MatMulLoweringPass::runOnOperation() {
    mlir::ConversionTarget target(getContext());
    mlir::RewritePatternSet patterns(&getContext());
    LowerToLLVMOptions llvmOptions(&getContext());
    LLVMTypeConverter typeConverter(&getContext(), llvmOptions);

    target.addLegalDialect<mlir::memref::MemRefDialect>();
    target.addLegalDialect<mlir::arith::ArithDialect>();
    target.addLegalDialect<mlir::scf::SCFDialect>();
    target.addLegalDialect<mlir::AffineDialect>();
    target.addLegalDialect<mlir::linalg::LinalgDialect>();
    target.addLegalDialect<mlir::LLVM::LLVMDialect>();
    target.addLegalDialect<mlir::math::MathDialect>();

    target.addLegalOp<mlir::daphne::ConvertDenseMatrixToMemRef>();
    target.addLegalOp<mlir::daphne::ConvertMemRefToDenseMatrix>();
    target.addLegalOp<mlir::daphne::DecRefOp>();

    target.addIllegalOp<mlir::daphne::MatMulOp>();

    patterns.insert<MatMulLowering>(&getContext());
    auto module = getOperation();
    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

std::unique_ptr<mlir::Pass> mlir::daphne::createMatMulOpLoweringPass() {
    return std::make_unique<MatMulLoweringPass>();
}
