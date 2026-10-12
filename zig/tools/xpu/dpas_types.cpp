// Sonde du débit des DPAS de Xe2 selon le type (fp16, bf16, int8, int4, int2), générée par nGEN : un thread SIMD16
// enchaîne `iters` fois 8 DPAS 8×8 sur 8 accumulateurs indépendants (registres 32 … 95), A et B fixes. Le K d'un DPAS
// est 8 × (éléments par mot de 32 bits) : 16, 32, 64, 128. Contrôle : A vaut 1 partout, B 1.0 (flottants) ou −1
// (entiers), la première ligne du premier accumulateur est écrite dans out : iters · K, ou −iters · K.
//
//   g++ -std=c++17 -O1 -fno-operator-names -I<ngen> dpas_types.cpp -o dpas_types && ./dpas_types SORTIE_DIR
#include <cstdio>
#include <fstream>
#include <string>

#include "ngen_elf.hpp"

using namespace ngen;

template <HW hw>
class DpasLoop : public ELFCodeGenerator<hw> {
    NGEN_FORWARD_ELF(hw)

public:
    DpasLoop(Product product, DataType acc, DataType ab, uint32_t a, uint32_t b) : ELFCodeGenerator<hw>(product) {
        newArgument("out", ExternalArgumentType::GlobalPtr);
        newArgument("iters", DataType::d);
        externalName("dpas_loop");
        requireSIMD(16);
        requireGRF(128);
        requireLocalID(1);
        requireDPAS();
        finalizeInterface();

        setDefaultNoMask();
        setDefaultAutoSWSB(true);
        prologue();

        const GRF B = r100, A = r110, cnt = r120;
        for (int i = 0; i < 8; i++) mov(16, GRF(B.getBase() + i).ud(), b);
        for (int i = 0; i < 4; i++) mov(16, GRF(A.getBase() + i).ud(), a);
        for (int j = 0; j < 64; j++) mov(16, GRF(32 + j).ud(), 0);
        mov(1, cnt.d(0), getArgument("iters"));
        Label top;
        mark(top);
        for (int j = 0; j < 8; j++) {
            const GRF c = GRF(32 + 8 * j);
            dpas(16, 8, 8, c.retype(acc), c.retype(acc), B.retype(ab), A.retype(ab));
        }
        add(1, cnt.d(0), cnt.d(0), -1);
        cmp(1 | gt | f0[0], cnt.d(0), 0);
        jmpi(1 | f0[0], top);
        const GRF addr = r121;                         // contrôle : la première ligne du premier accumulateur
        mov(1, addr.uq(0), getArgument("out"));
        store.ugm(1, block(DataSizeLSC::D32, 16), A64, addr, GRF(32));
        epilogue();
    }
};

int main(int argc, char **argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage : %s SORTIE_DIR\n", argv[0]);
        return 2;
    }
    const Product p(ProductFamily::GenericXe2, 0, PlatformType::Discrete);
    const struct {
        const char *name;
        DataType acc, ab;
        uint32_t a, b;                                 // A : uns ; B : 1.0 (flottants) ou −1 (entiers)
    } types[] = {{"hf", DataType::f, DataType::hf, 0x3C003C00, 0x3C003C00},
                 {"bf", DataType::f, DataType::bf, 0x3F803F80, 0x3F803F80},
                 {"b", DataType::d, DataType::b, 0x01010101, 0xFFFFFFFF},
                 {"s4", DataType::d, DataType::s4, 0x11111111, 0xFFFFFFFF},
                 {"s2", DataType::d, DataType::s2, 0x55555555, 0xFFFFFFFF}};
    for (auto &t : types) {
        auto bin = DpasLoop<HW::Xe2>(p, t.acc, t.ab, t.a, t.b).getBinary();
        const std::string path = std::string(argv[1]) + "/dpas_" + t.name + ".zebin";
        std::ofstream(path, std::ios::binary).write(reinterpret_cast<const char *>(bin.data()), bin.size());
        std::printf("%s : %zu octets\n", path.c_str(), bin.size());
    }
    return 0;
}
