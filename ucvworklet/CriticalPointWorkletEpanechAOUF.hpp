#ifndef UCV_CRITICAL_POINT_EPANECH_AOUF_h
#define UCV_CRITICAL_POINT_EPANECH_AOUF_h

#include <vtkm/worklet/WorkletPointNeighborhood.h>

#ifdef USE_LOG
#define LOG(x) x
#else
#define LOG(x)
#endif

// Epanech kernel and also avoid the over/under flow
struct CriticalPointWorkletEpanechAOUF : public vtkm::worklet::WorkletPointNeighborhood
{
public:
    CriticalPointWorkletEpanechAOUF(){};

    using ControlSignature = void(CellSetIn, FieldInNeighborhood, FieldInNeighborhood, FieldOut);

    using ExecutionSignature = void(_2, _3, _4, Boundary, WorkIndex);

    template <typename InPointField, typename OutPointField>
    VTKM_EXEC void operator()(const InPointField &minValue,
                              const InPointField &maxValue,
                              OutPointField &minProb,
                              const vtkm::exec::BoundaryState &boundary,
                              vtkm::Id LOG(WorkIndex)) const
    {
        // resluts is the coordinates of three dims
        auto minIndices = boundary.MinNeighborIndices(1);
        auto maxIndices = boundary.MaxNeighborIndices(1);

        // minIndices is supposed to be -1
        // maxIndices is supposed to be 1
        // if (WorkIndex == 0)
        //{
        // debug
        LOG(printf("workIndex is %d\n", WorkIndex));
        // printf("min index %d %d %d\n", minIndices[0], minIndices[1], minIndices[2]);
        // printf("max index %d %d %d\n", maxIndices[0], maxIndices[1], maxIndices[2]);

        // filter out the element in the boundry
        // if the element is at the boundry places its min prob is 0
        if ((maxIndices[0] - minIndices[0] < 2) || (maxIndices[1] - minIndices[1] < 2))
        {
            // if x and y is at the boundry, do not consider it
            minProb = 0;
            return;
        }

        vtkm::FloatDefault a1 = minValue.Get(0, 0, 0);
        vtkm::FloatDefault b1 = maxValue.Get(0, 0, 0);

        vtkm::FloatDefault a2 = minValue.Get(0, 1, 0);
        vtkm::FloatDefault b2 = maxValue.Get(0, 1, 0);

        vtkm::FloatDefault a3 = minValue.Get(0, -1, 0);
        vtkm::FloatDefault b3 = maxValue.Get(0, -1, 0);

        vtkm::FloatDefault a4 = minValue.Get(1, 0, 0);
        vtkm::FloatDefault b4 = maxValue.Get(1, 0, 0);

        vtkm::FloatDefault a5 = minValue.Get(-1, 0, 0);
        vtkm::FloatDefault b5 = maxValue.Get(-1, 0, 0);

        LOG(printf("check input a1 %f b1 %f a2 %f b2 %f a3 %f b3 %f a4 %f b4 %f a5 %f b5 %f\n", a1, b1, a2, b2, a3, b3, a4, b4, a5, b5));

        // filter out regions with zero values in it
        if (abs(a2 - 0.0) < 0.0000001 || abs(a3 - 0.0) < 0.0000001 || abs(a4 - 0.0) < 0.0000001 || abs(a5 - 0.0) < 0.0000001)
        {
            minProb = 0.0;
            return;
        }

        // if it is not under the region with zeros
        // do the preprocessing to avoid the data overflow
        // offset by a1 and scale it a little bit
        vtkm::FloatDefault a1N = (a1 - a1) * this->m_ScaleNum;
        vtkm::FloatDefault b1N = (b1 - a1) * this->m_ScaleNum;

        vtkm::FloatDefault a2N = (a2 - a1) * this->m_ScaleNum;
        vtkm::FloatDefault b2N = (b2 - a1) * this->m_ScaleNum;

        vtkm::FloatDefault a3N = (a3 - a1) * this->m_ScaleNum;
        vtkm::FloatDefault b3N = (b3 - a1) * this->m_ScaleNum;

        vtkm::FloatDefault a4N = (a4 - a1) * this->m_ScaleNum;
        vtkm::FloatDefault b4N = (b4 - a1) * this->m_ScaleNum;

        vtkm::FloatDefault a5N = (a5 - a1) * this->m_ScaleNum;
        vtkm::FloatDefault b5N = (b5 - a1) * this->m_ScaleNum;

        LOG(printf("check transformed input a1 %f b1 %f a2 %f b2 %f a3 %f b3 %f a4 %f b4 %f a5 %f b5 %f\n", a1N, b1N, a2N, b2N, a3N, b3N, a4N, b4N, a5N, b5N));

        // the logic in superOptimizedAnlyticalLocalMinimumProbabilityComputationEpanechnikov:

        // compute bmin
        vtkm::FloatDefault bMin = vtkm::Min(b1N, vtkm::Min(b2N, vtkm::Min(b3N, vtkm::Min(b4N, b5N))));
        LOG(printf("bmin %f\n", bMin));

        if (bMin <= a1N)
        {
            minProb = 0.0;
            return;
        }

        // startPointList = [a1, a2, a3, a4, a5]
        // order = np.argsort(startPointList)
        // interval, first is value second is actual index from 0 to 4
        vtkm::Vec<vtkm::Pair<vtkm::Float64, vtkm::Float64>, 5> interval;
        interval[0] = {a1N, b1N};
        interval[1] = {a2N, b2N};
        interval[2] = {a3N, b3N};
        interval[3] = {a4N, b4N};
        interval[4] = {a5N, b5N};
        // vtkm::Vec<vtkm::Id, 5> sotedIndex = ArgSort<5>(interval);
        // sort the arg and associated value together
        ArgSort<5>(interval);
        LOG(printf("sorted a [%f %f %f %f %f]\n", interval[0].first, interval[1].first, interval[2].first, interval[3].first, interval[4].first));

        // find interval contain bMin
        // the interval vector is sorted now
        vtkm::Id tartgetIndex;
        for (tartgetIndex = 4; tartgetIndex > 0; tartgetIndex--)
        {
            auto intervalFromEnd = interval[tartgetIndex];
            if ((bMin >= intervalFromEnd.first) && (bMin <= intervalFromEnd.second))
            {
                break;
            }
        }
        LOG(printf("---endInterval %d\n", tartgetIndex));

        // create x1 limit, init it as negative value
        vtkm::Vec<vtkm::Float64, 6> x1Limit(vtkm::Nan64());
        vtkm::Id index;
        for (index = 0; index < tartgetIndex + 1; index++)
        {
            x1Limit[index] = interval[index].first;
        }
        // add bMin as the last element
        x1Limit[index] = bMin;

        // find the a1
        vtkm::Id indexa1 = -1;
        for (vtkm::Id i = 0; i < 6; i++)
        {
            if (vtkm::Abs(x1Limit[i] - a1N) < 0.0000001)
            {
                indexa1 = i;
                break;
            }
        }

        if (indexa1 == -1)
        {
            // run time error
            printf("error, the indexa1 is not supposed to be -1");
            minProb = 0.0;
            return;
        }

        vtkm::FloatDefault w1 = b1N - a1N;
        vtkm::FloatDefault m1 = (b1N + a1N) / 2.0;
        // call SuperOptimizedCaseEpanechnikov
        minProb = SuperOptimizedCaseEpanechnikov(indexa1, x1Limit, interval, w1, m1);
        return;
    }
    // ascending
    template <vtkm::Id Size>
    VTKM_EXEC inline void ArgSort(vtkm::Vec<vtkm::Pair<vtkm::Float64, vtkm::Float64>, Size> &interval) const
    {
        for (int i = 0; i < Size; i++)
        {
            for (int j = 0; j < Size - i - 1; j++)
            {
                // compare element i and j
                if (interval[j].first > interval[j + 1].first)
                {
                    // swap
                    auto temp = interval[j];
                    interval[j] = interval[j + 1];
                    interval[j + 1] = temp;
                }
            }
        }
        return;
    }
    //        minimaProb = superOptimizedCase(indexOfa1,x1Limits, sortedI, w1)
    VTKM_EXEC inline vtkm::Float64 SuperOptimizedCaseEpanechnikov(vtkm::Id indexOfa1,
                                                                  vtkm::Vec<vtkm::Float64, 6> x1Limits,
                                                                  vtkm::Vec<vtkm::Pair<vtkm::Float64, vtkm::Float64>, 5> interval,
                                                                  vtkm::Float64 w1, vtkm::Float64 m1) const
    {
        // TODO, this function need to be updated according to the python code
        LOG(printf("---debug SuperOptimizedCaseEpanechnikov indexOfa1 %d w1 %f m1 %f\n", indexOfa1, w1, m1));
        LOG(printf("---debug x1Limits\n"));
        for (int i = 0; i < 6; i++)
        {
            LOG(printf("%f ", x1Limits[i]));
        }
        LOG(printf("\n"));
        LOG(printf("---debug interval\n"));
        for (int i = 0; i < 5; i++)
        {
            LOG(printf("[%f %f]", interval[i].first, interval[i].second));
        }
        LOG(printf("\n"));
        vtkm::Float64 minProb = 0;

        vtkm::Vec<vtkm::Float64, 6> upLimits({0, vtkm::Nan64(), vtkm::Nan64(), vtkm::Nan64(), vtkm::Nan64(), vtkm::Nan64()});
        vtkm::Vec<vtkm::Float64, 6> normalizerFactors({0, 3.0 / (2.0 * w1), 1, 1, 1, 1});
        // midpoints = [0, m1, None, None, None, None]
        // intervalHalfWidths = [0, w1/2, None,None,None,None]
        vtkm::Vec<vtkm::Float64, 6> midpoints({0, m1, vtkm::Nan64(), vtkm::Nan64(), vtkm::Nan64(), vtkm::Nan64()});
        vtkm::Vec<vtkm::Float64, 6> intervalHalfWidths({0, w1 / 2.0, vtkm::Nan64(), vtkm::Nan64(), vtkm::Nan64(), vtkm::Nan64()});

        for (vtkm::Id k = 0; k < indexOfa1; k++)
        {
            auto tempInterval = interval[k];
            upLimits[k + 2] = tempInterval.second;
            normalizerFactors[k + 2] = 3.0 / (2.0 * (tempInterval.second - tempInterval.first));
            intervalHalfWidths[k + 2] = (tempInterval.second - tempInterval.first) / 2.0;
            midpoints[k + 2] = (tempInterval.second + tempInterval.first) / 2.0;
        }

        // computing cases before indexOfa1
        minProb = minProb + computeIntegralEpanechnikov(
                                x1Limits[indexOfa1], x1Limits[indexOfa1 + 1],
                                upLimits[2], upLimits[3], upLimits[4], upLimits[5],
                                normalizerFactors[1], normalizerFactors[2], normalizerFactors[3], normalizerFactors[4], normalizerFactors[5],
                                midpoints[1], midpoints[2], midpoints[3], midpoints[4], midpoints[5],
                                intervalHalfWidths[1], intervalHalfWidths[2], intervalHalfWidths[3], intervalHalfWidths[4], intervalHalfWidths[5]);

        LOG(printf("----debug minProb first case %f\n", minProb));
        // computing cases after indexOfa1
        for (vtkm::Id i = indexOfa1 + 1; i < 5; i++)
        {
            // break if x1Limits goes to nan
            if (x1Limits[i] == vtkm::Nan64())
            {
                break;
            }
            auto tempInterval = interval[i];
            upLimits[i + 1] = tempInterval.second;
            normalizerFactors[i + 1] = 3.0 / (2.0 * (tempInterval.second - tempInterval.first));
            intervalHalfWidths[i + 1] = (tempInterval.second - tempInterval.first) / 2.0;
            midpoints[i + 1] = (tempInterval.second + tempInterval.first) / 2.0;
            minProb = minProb + computeIntegralEpanechnikov(
                                    x1Limits[i], x1Limits[i + 1],
                                    upLimits[2], upLimits[3], upLimits[4], upLimits[5],
                                    normalizerFactors[1], normalizerFactors[2], normalizerFactors[3], normalizerFactors[4], normalizerFactors[5],
                                    midpoints[1], midpoints[2], midpoints[3], midpoints[4], midpoints[5],
                                    intervalHalfWidths[1], intervalHalfWidths[2], intervalHalfWidths[3], intervalHalfWidths[4], intervalHalfWidths[5]);
        }
        return minProb;
    }

    VTKM_EXEC inline vtkm::Float64 computeIntegralEpanechnikov(vtkm::Float64 l,
                                                               vtkm::Float64 h,
                                                               vtkm::Float64 h2,
                                                               vtkm::Float64 h3,
                                                               vtkm::Float64 h4,
                                                               vtkm::Float64 h5,
                                                               vtkm::Float64 n1,
                                                               vtkm::Float64 n2,
                                                               vtkm::Float64 n3,
                                                               vtkm::Float64 n4,
                                                               vtkm::Float64 n5,
                                                               vtkm::Float64 mid1,
                                                               vtkm::Float64 mid2,
                                                               vtkm::Float64 mid3,
                                                               vtkm::Float64 mid4,
                                                               vtkm::Float64 mid5,
                                                               vtkm::Float64 wid1,
                                                               vtkm::Float64 wid2,
                                                               vtkm::Float64 wid3,
                                                               vtkm::Float64 wid4,
                                                               vtkm::Float64 wid5) const
    {
        LOG(printf("---debug ComputeIntegral input %f %f %f %f %f %f, %f %f %f %f %f, %f %f %f %f %f, %f %f %f %f %f\n",
                   l, h, h2, h3, h4, h5, n1, n2, n3, n4, n5, mid1, mid2, mid3, mid4, mid5, wid1, wid2, wid3, wid4, wid5));

        vtkm::Float64 intUp = 0.0;
        vtkm::Float64 intDown = 0.0;

        vtkm::Float64 normalizingFactor = n1 * n2 * n3 * n4 * n5;

        // ln is none
        bool ln = isnan(l);
        bool hn = isnan(h);

        bool h2n = isnan(h2);
        bool h3n = isnan(h3);
        bool h4n = isnan(h4);
        bool h5n = isnan(h5);

        bool n1n = isnan(n1);
        bool n2n = isnan(n2);
        bool n3n = isnan(n3);
        bool n4n = isnan(n4);
        bool n5n = isnan(n5);

        // integral 1
        if ((!ln) && (!hn) && (h2n) && (h3n) && (h4n) && (h5n))
        {
            intUp = normalizingFactor * (h - vtkm::Pow((h - mid1), 3) / (3 * wid1 * wid1));
            intDown = normalizingFactor * (l - vtkm::Pow((l - mid1), 3) / (3 * wid1 * wid1));
            if ((intUp - intDown) > 1)
                printf("error in integral 1");
        }
        // integral 2
        if ((!ln) && (!hn) && (!h2n) && (h3n) && (h4n) && (h5n))
        {
            vtkm::Float64 k2 = h2 - (vtkm::Pow((h2 - mid2), 3) / (3 * (vtkm::Pow(wid2, 2))));
            intUp = normalizingFactor * (h * (-15 * mid1 * mid1 * (12 * k2 * (wid2)*wid2 - 4 * vtkm::Pow(mid2, 3) + 6 * mid2 * mid2 * h - 4 * mid2 * h * h - 6 * h * (wid2)*wid2 + h * h * h) + 6 * mid1 * h * (30 * k2 * (wid2)*wid2 - 10 * mid2 * mid2 * mid2 + 20 * mid2 * mid2 * h - 15 * mid2 * h * h + 4 * h * (h * h - 5 * (wid2)*wid2)) + 5 * (-12 * k2 * (wid2)*wid2 * (h * h - 3 * (wid1)*wid1) + 3 * h * (wid1)*wid1 * (h * h - 6 * (wid2)*wid2) + 9 * h * h * h * (wid2)*wid2 - 2 * vtkm::Pow(h, 5)) + 20 * mid2 * mid2 * mid2 * (h * h - 3 * (wid1)*wid1) - 45 * mid2 * mid2 * (h * h * h - 2 * h * (wid1)*wid1) + mid2 * (36 * vtkm::Pow(h, 4) - 60 * h * h * (wid1)*wid1))) / (180 * (wid1)*wid1 * (wid2)*wid2);
            intDown = normalizingFactor * (l * (-15 * mid1 * mid1 * (12 * k2 * (wid2)*wid2 - 4 * vtkm::Pow(mid2, 3) + 6 * mid2 * mid2 * l - 4 * mid2 * l * l - 6 * l * (wid2)*wid2 + l * l * l) + 6 * mid1 * l * (30 * k2 * (wid2)*wid2 - 10 * mid2 * mid2 * mid2 + 20 * mid2 * mid2 * l - 15 * mid2 * l * l + 4 * l * (l * l - 5 * (wid2)*wid2)) + 5 * (-12 * k2 * (wid2)*wid2 * (l * l - 3 * (wid1)*wid1) + 3 * l * (wid1)*wid1 * (l * l - 6 * (wid2)*wid2) + 9 * l * l * l * (wid2)*wid2 - 2 * vtkm::Pow(l, 5)) + 20 * mid2 * mid2 * mid2 * (l * l - 3 * (wid1)*wid1) - 45 * mid2 * mid2 * (l * l * l - 2 * l * (wid1)*wid1) + mid2 * (36 * vtkm::Pow(l, 4) - 60 * l * l * (wid1)*wid1))) / (180 * (wid1)*wid1 * (wid2)*wid2);

            if ((intUp - intDown) > 1)
                printf("error in integral 2");
        }
        // integral 3
        if ((!ln) && (!hn) && (!h2n) && (!h3n) && (h4n) && (h5n))
        {
            vtkm::Float64 k2 = h2 - vtkm::Pow((h2 - mid2), 3) / (3 * (wid2 * wid2));
            vtkm::Float64 k3 = h3 - vtkm::Pow((h3 - mid3), 3) / (3 * (wid3 * wid3));

            vtkm::Float64 f = (-1) * (1.0 / (22680 * wid1 * wid1 * wid2 * wid2 * wid3 * wid3));
            vtkm::Float64 f9 = f * 280;
            vtkm::Float64 f8 = f * (-945 * mid3 - 945 * mid2 - 630 * mid1);
            vtkm::Float64 f7 = f * (-1080 * wid3 * wid3 - 1080 * wid2 * wid2 - 360 * wid1 * wid1 + 1080 * mid3 * mid3 + (3240 * mid2 + 2160 * mid1) * mid3 + 1080 * mid2 * mid2 + 2160 * mid1 * mid2 + 360 * mid1 * mid1);
            vtkm::Float64 f6 = f * ((3780 * mid2 + 2520 * mid1 + 1260 * k3) * wid3 * wid3 + (3780 * mid3 + 2520 * mid1 + 1260 * k2) * wid2 * wid2 + (1260 * mid3 + 1260 * mid2) * wid1 * wid1 - 420 * mid3 * mid3 * mid3 + (-3780 * mid2 - 2520 * mid1) * mid3 * mid3 + (-3780 * mid2 * mid2 - 7560 * mid1 * mid2 - 1260 * mid1 * mid1) * mid3 - 420 * mid2 * mid2 * mid2 - 2520 * mid1 * mid2 * mid2 - 1260 * mid1 * mid1 * mid2);
            vtkm::Float64 f5 = f * ((4536 * wid2 * wid2 + 1512 * wid1 * wid1 - 4536 * mid2 * mid2 + (-9072 * mid1 - 4536 * k3) * mid2 - 1512 * mid1 * mid1 - 3024 * k3 * mid1) * wid3 * wid3 + (1512 * wid1 * wid1 - 4536 * mid3 * mid3 + (-9072 * mid1 - 4536 * k2) * mid3 - 1512 * mid1 * mid1 - 3024 * k2 * mid1) * wid2 * wid2 + (-1512 * mid3 * mid3 - 4536 * mid2 * mid3 - 1512 * mid2 * mid2) * wid1 * wid1 + (1512 * mid2 + 1008 * mid1) * mid3 * mid3 * mid3 + (4536 * mid2 * mid2 + 9072 * mid1 * mid2 + 1512 * mid1 * mid1) * mid3 * mid3 + (1512 * mid2 * mid2 * mid2 + 9072 * mid1 * mid2 * mid2 + 4536 * mid1 * mid1 * mid2) * mid3 + 1008 * mid1 * mid2 * mid2 * mid2 + 1512 * mid1 * mid1 * mid2 * mid2);
            vtkm::Float64 f4 = f * (((-11340 * mid1 - 5670 * k3 - 5670 * k2) * wid2 * wid2 + (-5670 * mid2 - 1890 * k3) * wid1 * wid1 + 1890 * mid2 * mid2 * mid2 + (11340 * mid1 + 5670 * k3) * mid2 * mid2 + (5670 * mid1 * mid1 + 11340 * k3 * mid1) * mid2 + 1890 * k3 * mid1 * mid1) * wid3 * wid3 + ((-5670 * mid3 - 1890 * k2) * wid1 * wid1 + 1890 * mid3 * mid3 * mid3 + (11340 * mid1 + 5670 * k2) * mid3 * mid3 + (5670 * mid1 * mid1 + 11340 * k2 * mid1) * mid3 + 1890 * k2 * mid1 * mid1) * wid2 * wid2 + (630 * mid3 * mid3 * mid3 + 5670 * mid2 * mid3 * mid3 + 5670 * mid2 * mid2 * mid3 + 630 * mid2 * mid2 * mid2) * wid1 * wid1 + (-1890 * mid2 * mid2 - 3780 * mid1 * mid2 - 630 * mid1 * mid1) * mid3 * mid3 * mid3 + (-1890 * mid2 * mid2 * mid2 - 11340 * mid1 * mid2 * mid2 - 5670 * mid1 * mid1 * mid2) * mid3 * mid3 + (-3780 * mid1 * mid2 * mid2 * mid2 - 5670 * mid1 * mid1 * mid2 * mid2) * mid3 - 630 * mid1 * mid1 * mid2 * mid2 * mid2);
            vtkm::Float64 f3 = f * (((-7560 * wid1 * wid1 + 7560 * mid1 * mid1 + (15120 * k3 + 15120 * k2) * mid1 + 7560 * k2 * k3) * wid2 * wid2 + (7560 * mid2 * mid2 + 7560 * k3 * mid2) * wid1 * wid1 + (-5040 * mid1 - 2520 * k3) * mid2 * mid2 * mid2 + (-7560 * mid1 * mid1 - 15120 * k3 * mid1) * mid2 * mid2 - 7560 * k3 * mid1 * mid1 * mid2) * wid3 * wid3 + ((7560 * mid3 * mid3 + 7560 * k2 * mid3) * wid1 * wid1 + (-5040 * mid1 - 2520 * k2) * mid3 * mid3 * mid3 + (-7560 * mid1 * mid1 - 15120 * k2 * mid1) * mid3 * mid3 - 7560 * k2 * mid1 * mid1 * mid3) * wid2 * wid2 + (-2520 * mid2 * mid3 * mid3 * mid3 - 7560 * mid2 * mid2 * mid3 * mid3 - 2520 * mid2 * mid2 * mid2 * mid3) * wid1 * wid1 + (840 * mid2 * mid2 * mid2 + 5040 * mid1 * mid2 * mid2 + 2520 * mid1 * mid1 * mid2) * mid3 * mid3 * mid3 + (5040 * mid1 * mid2 * mid2 * mid2 + 7560 * mid1 * mid1 * mid2 * mid2) * mid3 * mid3 + 2520 * mid1 * mid1 * mid2 * mid2 * mid2 * mid3);
            vtkm::Float64 f2 = f * ((((11340 * k3 + 11340 * k2) * wid1 * wid1 + (-11340 * k3 - 11340 * k2) * mid1 * mid1 - 22680 * k2 * k3 * mid1) * wid2 * wid2 + (-3780 * mid2 * mid2 * mid2 - 11340 * k3 * mid2 * mid2) * wid1 * wid1 + (3780 * mid1 * mid1 + 7560 * k3 * mid1) * mid2 * mid2 * mid2 + 11340 * k3 * mid1 * mid1 * mid2 * mid2) * wid3 * wid3 + ((-3780 * mid3 * mid3 * mid3 - 11340 * k2 * mid3 * mid3) * wid1 * wid1 + (3780 * mid1 * mid1 + 7560 * k2 * mid1) * mid3 * mid3 * mid3 + 11340 * k2 * mid1 * mid1 * mid3 * mid3) * wid2 * wid2 + (3780 * mid2 * mid2 * mid3 * mid3 * mid3 + 3780 * mid2 * mid2 * mid2 * mid3 * mid3) * wid1 * wid1 + (-2520 * mid1 * mid2 * mid2 * mid2 - 3780 * mid1 * mid1 * mid2 * mid2) * mid3 * mid3 * mid3 - 3780 * mid1 * mid1 * mid2 * mid2 * mid2 * mid3 * mid3);
            vtkm::Float64 f1 = f * (((22680 * k2 * k3 * mid1 * mid1 - 22680 * k2 * k3 * wid1 * wid1) * wid2 * wid2 + 7560 * k3 * mid2 * mid2 * mid2 * wid1 * wid1 - 7560 * k3 * mid1 * mid1 * mid2 * mid2 * mid2) * wid3 * wid3 + (7560 * k2 * mid3 * mid3 * mid3 * wid1 * wid1 - 7560 * k2 * mid1 * mid1 * mid3 * mid3 * mid3) * wid2 * wid2 - 2520 * mid2 * mid2 * mid2 * mid3 * mid3 * mid3 * wid1 * wid1 + 2520 * mid1 * mid1 * mid2 * mid2 * mid2 * mid3 * mid3 * mid3);

            intUp = normalizingFactor * (f9 * vtkm::Pow(h, 9) + f8 * vtkm::Pow(h, 8) + f7 * vtkm::Pow(h, 7) + f6 * vtkm::Pow(h, 6) + f5 * vtkm::Pow(h, 5) + f4 * vtkm::Pow(h, 4) + f3 * vtkm::Pow(h, 3) + f2 * h * h + f1 * h);
            intDown = normalizingFactor * (f9 * vtkm::Pow(l, 9) + f8 * vtkm::Pow(l, 8) + f7 * vtkm::Pow(l, 7) + f6 * vtkm::Pow(l, 6) + f5 * vtkm::Pow(l, 5) + f4 * vtkm::Pow(l, 4) + f3 * vtkm::Pow(l, 3) + f2 * l * l + f1 * l);

            if ((intUp - intDown) > 1)
            {
                printf("error in integral 3\n");
            }
        }
        // integral 4
        if ((!ln) && (!hn) && (!h2n) && (!h3n) && (!h4n) && (h5n))
        {

            vtkm::Float64 k2 = h2 - (vtkm::Pow((h2 - mid2), 3) / (3 * (wid2 * wid2)));
            vtkm::Float64 k3 = h3 - (vtkm::Pow((h3 - mid3), 3) / (3 * (wid3 * wid3)));
            vtkm::Float64 k4 = h4 - (vtkm::Pow((h4 - mid4), 3) / (3 * (wid4 * wid4)));

            vtkm::Float64 f = 1.0 / (9.0 * wid1 * wid1 * wid2 * wid2 * wid3 * wid3);

            vtkm::Float64 f8 = f * (-1);
            vtkm::Float64 f7 = f * (2 * mid1 + 3 * mid2 + 3 * mid3);
            vtkm::Float64 f6 = f * (-mid1 * mid1 - 6 * mid2 * mid1 - 6 * mid3 * mid1 - 3 * mid2 * mid2 - 3 * mid3 * mid3 + wid1 * wid1 + 3 * wid2 * wid2 + 3 * wid3 * wid3 - 9 * mid2 * mid3);
            vtkm::Float64 f5 = f * (mid2 * mid2 * mid2 + 6 * mid1 * mid2 * mid2 + 9 * mid3 * mid2 * mid2 + 3 * mid1 * mid1 * mid2 + 9 * mid3 * mid3 * mid2 - 3 * wid1 * wid1 * mid2 - 9 * wid3 * wid3 * mid2 + 18 * mid1 * mid3 * mid2 + mid3 * mid3 * mid3 + 6 * mid1 * mid3 * mid3 - 3 * mid3 * wid1 * wid1 - 3 * k2 * wid2 * wid2 - 6 * mid1 * wid2 * wid2 - 9 * mid3 * wid2 * wid2 - 3 * k3 * wid3 * wid3 - 6 * mid1 * wid3 * wid3 + 3 * mid1 * mid1 * mid3);
            vtkm::Float64 f4 = f * (-2 * mid1 * mid2 * mid2 * mid2 - 3 * mid3 * mid2 * mid2 * mid2 - 3 * mid1 * mid1 * mid2 * mid2 - 9 * mid3 * mid3 * mid2 * mid2 + 3 * wid1 * wid1 * mid2 * mid2 + 9 * wid3 * wid3 * mid2 * mid2 - 18 * mid1 * mid3 * mid2 * mid2 - 3 * mid3 * mid3 * mid3 * mid2 - 18 * mid1 * mid3 * mid3 * mid2 + 9 * mid3 * wid1 * wid1 * mid2 + 9 * k3 * wid3 * wid3 * mid2 + 18 * mid1 * wid3 * wid3 * mid2 - 9 * mid1 * mid1 * mid3 * mid2 - 2 * mid1 * mid3 * mid3 * mid3 - 3 * mid1 * mid1 * mid3 * mid3 + 3 * mid3 * mid3 * wid1 * wid1 + 3 * mid1 * mid1 * wid2 * wid2 + 9 * mid3 * mid3 * wid2 * wid2 - 3 * wid1 * wid1 * wid2 * wid2 + 6 * k2 * mid1 * wid2 * wid2 + 9 * k2 * mid3 * wid2 * wid2 + 18 * mid1 * mid3 * wid2 * wid2 + 3 * mid1 * mid1 * wid3 * wid3 - 3 * wid1 * wid1 * wid3 * wid3 - 9 * wid2 * wid2 * wid3 * wid3 + 6 * k3 * mid1 * wid3 * wid3);
            vtkm::Float64 f3 = f * (mid1 * mid1 * mid2 * mid2 * mid2 + 3 * mid3 * mid3 * mid2 * mid2 * mid2 - wid1 * wid1 * mid2 * mid2 * mid2 - 3 * wid3 * wid3 * mid2 * mid2 * mid2 + 6 * mid1 * mid3 * mid2 * mid2 * mid2 + 3 * mid3 * mid3 * mid3 * mid2 * mid2 + 18 * mid1 * mid3 * mid3 * mid2 * mid2 - 9 * mid3 * wid1 * wid1 * mid2 * mid2 - 9 * k3 * wid3 * wid3 * mid2 * mid2 - 18 * mid1 * wid3 * wid3 * mid2 * mid2 + 9 * mid1 * mid1 * mid3 * mid2 * mid2 + 6 * mid1 * mid3 * mid3 * mid3 * mid2 + 9 * mid1 * mid1 * mid3 * mid3 * mid2 - 9 * mid3 * mid3 * wid1 * wid1 * mid2 - 9 * mid1 * mid1 * wid3 * wid3 * mid2 + 9 * wid1 * wid1 * wid3 * wid3 * mid2 - 18 * k3 * mid1 * wid3 * wid3 * mid2 + mid1 * mid1 * mid3 * mid3 * mid3 - mid3 * mid3 * mid3 * wid1 * wid1 - 3 * mid3 * mid3 * mid3 * wid2 * wid2 - 3 * k2 * mid1 * mid1 * wid2 * wid2 - 9 * k2 * mid3 * mid3 * wid2 * wid2 - 18 * mid1 * mid3 * mid3 * wid2 * wid2 + 3 * k2 * wid1 * wid1 * wid2 * wid2 + 9 * mid3 * wid1 * wid1 * wid2 * wid2 - 9 * mid1 * mid1 * mid3 * wid2 * wid2 - 18 * k2 * mid1 * mid3 * wid2 * wid2 - 3 * k3 * mid1 * mid1 * wid3 * wid3 + 3 * k3 * wid1 * wid1 * wid3 * wid3 + 9 * k2 * wid2 * wid2 * wid3 * wid3 + 9 * k3 * wid2 * wid2 * wid3 * wid3 + 18 * mid1 * wid2 * wid2 * wid3 * wid3);
            vtkm::Float64 f2 = f * (-mid3 * mid3 * mid3 * mid2 * mid2 * mid2 - 6 * mid1 * mid3 * mid3 * mid2 * mid2 * mid2 + 3 * mid3 * wid1 * wid1 * mid2 * mid2 * mid2 + 3 * k3 * wid3 * wid3 * mid2 * mid2 * mid2 + 6 * mid1 * wid3 * wid3 * mid2 * mid2 * mid2 - 3 * mid1 * mid1 * mid3 * mid2 * mid2 * mid2 - 6 * mid1 * mid3 * mid3 * mid3 * mid2 * mid2 - 9 * mid1 * mid1 * mid3 * mid3 * mid2 * mid2 + 9 * mid3 * mid3 * wid1 * wid1 * mid2 * mid2 + 9 * mid1 * mid1 * wid3 * wid3 * mid2 * mid2 - 9 * wid1 * wid1 * wid3 * wid3 * mid2 * mid2 + 18 * k3 * mid1 * wid3 * wid3 * mid2 * mid2 - 3 * mid1 * mid1 * mid3 * mid3 * mid3 * mid2 + 3 * mid3 * mid3 * mid3 * wid1 * wid1 * mid2 + 9 * k3 * mid1 * mid1 * wid3 * wid3 * mid2 - 9 * k3 * wid1 * wid1 * wid3 * wid3 * mid2 + 3 * k2 * mid3 * mid3 * mid3 * wid2 * wid2 + 6 * mid1 * mid3 * mid3 * mid3 * wid2 * wid2 + 9 * mid1 * mid1 * mid3 * mid3 * wid2 * wid2 + 18 * k2 * mid1 * mid3 * mid3 * wid2 * wid2 - 9 * mid3 * mid3 * wid1 * wid1 * wid2 * wid2 - 9 * k2 * mid3 * wid1 * wid1 * wid2 * wid2 + 9 * k2 * mid1 * mid1 * mid3 * wid2 * wid2 - 9 * mid1 * mid1 * wid2 * wid2 * wid3 * wid3 + 9 * wid1 * wid1 * wid2 * wid2 * wid3 * wid3 - 9 * k2 * k3 * wid2 * wid2 * wid3 * wid3 - 18 * k2 * mid1 * wid2 * wid2 * wid3 * wid3 - 18 * k3 * mid1 * wid2 * wid2 * wid3 * wid3);
            vtkm::Float64 f1 = f * (2 * mid1 * mid3 * mid3 * mid3 * mid2 * mid2 * mid2 + 3 * mid1 * mid1 * mid3 * mid3 * mid2 * mid2 * mid2 - 3 * mid3 * mid3 * wid1 * wid1 * mid2 * mid2 * mid2 - 3 * mid1 * mid1 * wid3 * wid3 * mid2 * mid2 * mid2 + 3 * wid1 * wid1 * wid3 * wid3 * mid2 * mid2 * mid2 - 6 * k3 * mid1 * wid3 * wid3 * mid2 * mid2 * mid2 + 3 * mid1 * mid1 * mid3 * mid3 * mid3 * mid2 * mid2 - 3 * mid3 * mid3 * mid3 * wid1 * wid1 * mid2 * mid2 - 9 * k3 * mid1 * mid1 * wid3 * wid3 * mid2 * mid2 + 9 * k3 * wid1 * wid1 * wid3 * wid3 * mid2 * mid2 - 3 * mid1 * mid1 * mid3 * mid3 * mid3 * wid2 * wid2 - 6 * k2 * mid1 * mid3 * mid3 * mid3 * wid2 * wid2 - 9 * k2 * mid1 * mid1 * mid3 * mid3 * wid2 * wid2 + 3 * mid3 * mid3 * mid3 * wid1 * wid1 * wid2 * wid2 + 9 * k2 * mid3 * mid3 * wid1 * wid1 * wid2 * wid2 + 9 * k2 * mid1 * mid1 * wid2 * wid2 * wid3 * wid3 + 9 * k3 * mid1 * mid1 * wid2 * wid2 * wid3 * wid3 - 9 * k2 * wid1 * wid1 * wid2 * wid2 * wid3 * wid3 - 9 * k3 * wid1 * wid1 * wid2 * wid2 * wid3 * wid3 + 18 * k2 * k3 * mid1 * wid2 * wid2 * wid3 * wid3);
            vtkm::Float64 f0 = f * (-mid1 * mid1 * mid2 * mid2 * mid2 * mid3 * mid3 * mid3 + mid2 * mid2 * mid2 * mid3 * mid3 * mid3 * wid1 * wid1 + 3 * k2 * mid1 * mid1 * mid3 * mid3 * mid3 * wid2 * wid2 - 3 * k2 * mid3 * mid3 * mid3 * wid1 * wid1 * wid2 * wid2 + 3 * k3 * mid1 * mid1 * mid2 * mid2 * mid2 * wid3 * wid3 - 3 * k3 * mid2 * mid2 * mid2 * wid1 * wid1 * wid3 * wid3 - 9 * k2 * k3 * mid1 * mid1 * wid2 * wid2 * wid3 * wid3 + 9 * k2 * k3 * wid1 * wid1 * wid2 * wid2 * wid3 * wid3);

            vtkm::Float64 s = 1.0 / (83160 * wid4 * wid4);
            vtkm::Float64 s12 = s * (2310 * f8);
            vtkm::Float64 s11 = s * (2520 * f7 - 7560 * f8 * mid4);
            vtkm::Float64 s10 = s * (-8316 * f8 * wid4 * wid4 + 8316 * f8 * mid4 * mid4 - 8316 * f7 * mid4 + 2772 * f6);
            vtkm::Float64 s9 = s * ((9240 * f8 * k4 - 9240 * f7) * wid4 * wid4 - 3080 * f8 * mid4 * mid4 + 9240 * f7 * mid4 * mid4 - 9240 * f6 * mid4 + 3080 * f5);
            vtkm::Float64 s8 = s * ((10395 * f7 * k4 - 10395 * f6) * wid4 * wid4 - 3465 * f7 * mid4 * mid4 + 10395 * f6 * mid4 * mid4 - 10395 * f5 * mid4 + 3465 * f4);
            vtkm::Float64 s7 = s * ((11880 * f6 * k4 - 11880 * f5) * wid4 * wid4 - 3960 * f6 * mid4 * mid4 + 11880 * f5 * mid4 * mid4 - 11880 * f4 * mid4 + 3960 * f3);
            vtkm::Float64 s6 = s * ((13860 * f5 * k4 - 13860 * f4) * wid4 * wid4 - 4620 * f5 * mid4 * mid4 + 13860 * f4 * mid4 * mid4 - 13860 * f3 * mid4 + 4620 * f2);
            vtkm::Float64 s5 = s * ((16632 * f4 * k4 - 16632 * f3) * wid4 * wid4 - 5544 * f4 * mid4 * mid4 + 16632 * f3 * mid4 * mid4 - 16632 * f2 * mid4 + 5544 * f1);
            vtkm::Float64 s4 = s * ((20790 * f3 * k4 - 20790 * f2) * wid4 * wid4 - 6930 * f3 * mid4 * mid4 + 20790 * f2 * mid4 * mid4 - 20790 * f1 * mid4 + 6930 * f0);
            vtkm::Float64 s3 = s * ((27720 * f2 * k4 - 27720 * f1) * wid4 * wid4 - 9240 * f2 * mid4 * mid4 + 27720 * f1 * mid4 * mid4 - 27720 * f0 * mid4);
            vtkm::Float64 s2 = s * ((41580 * f1 * k4 - 41580 * f0) * wid4 * wid4 - 13860 * f1 * mid4 * mid4 + 41580 * f0 * mid4 * mid4);
            vtkm::Float64 s1 = s * (83160 * f0 * k4 * wid4 * wid4 - 27720 * f0 * mid4 * mid4);

            intUp = normalizingFactor * (s12 * (vtkm::Pow(h, 12)) + s11 * (vtkm::Pow(h, 11)) + s10 * (vtkm::Pow(h, 10)) + s9 * (vtkm::Pow(h, 9)) + s8 * vtkm::Pow(h, 8) + s7 * vtkm::Pow(h, 7) + s6 * vtkm::Pow(h, 6) + s5 * vtkm::Pow(h, 5) + s4 * vtkm::Pow(h, 4) + s3 * vtkm::Pow(h, 3) + s2 * h * h + s1 * h);
            intDown = normalizingFactor * (s12 * (vtkm::Pow(l, 12)) + s11 * (vtkm::Pow(l, 11)) + s10 * (vtkm::Pow(l, 10)) + s9 * (vtkm::Pow(l, 9)) + s8 * vtkm::Pow(l, 8) + s7 * vtkm::Pow(l, 7) + s6 * vtkm::Pow(l, 6) + s5 * vtkm::Pow(l, 5) + s4 * vtkm::Pow(l, 4) + s3 * vtkm::Pow(l, 3) + s2 * l * l + s1 * l);

            if ((intUp - intDown) > 1)
            {
                printf("error in integral 4\n");
            }
        }
        // integral 5
        if ((!ln) && (!hn) && (!h2n) && (!h3n) && (!h4n) && (!h5n))
        {

            vtkm::Float64 k2 = h2 - (vtkm::Pow((h2 - mid2), 3) / (3 * (wid2 * wid2)));
            vtkm::Float64 k3 = h3 - (vtkm::Pow((h3 - mid3), 3) / (3 * (wid3 * wid3)));
            vtkm::Float64 k4 = h4 - (vtkm::Pow((h4 - mid4), 3) / (3 * (wid4 * wid4)));
            vtkm::Float64 k5 = h5 - (vtkm::Pow((h5 - mid5), 3) / (3 * (wid5 * wid5)));

            // Below is plain multiplication of (w1^2 - (x-m1)^2/(w1^2))*(k2 - x + (x-m2)^3/(3w2^2))*(k3 - x + (x-m3)^3/(3w3^2))
            // separated into factors f of x^8, x^7.....

            vtkm::Float64 f = 1.0 / (9.0 * wid1 * wid1 * wid2 * wid2 * wid3 * wid3);

            vtkm::Float64 f8 = f * (-1);
            vtkm::Float64 f7 = f * (2 * mid1 + 3 * mid2 + 3 * mid3);
            vtkm::Float64 f6 = f * (-mid1 * mid1 - 6 * mid2 * mid1 - 6 * mid3 * mid1 - 3 * mid2 * mid2 - 3 * mid3 * mid3 + wid1 * wid1 + 3 * wid2 * wid2 + 3 * wid3 * wid3 - 9 * mid2 * mid3);
            vtkm::Float64 f5 = f * (mid2 * mid2 * mid2 + 6 * mid1 * mid2 * mid2 + 9 * mid3 * mid2 * mid2 + 3 * mid1 * mid1 * mid2 + 9 * mid3 * mid3 * mid2 - 3 * wid1 * wid1 * mid2 - 9 * wid3 * wid3 * mid2 + 18 * mid1 * mid3 * mid2 + mid3 * mid3 * mid3 + 6 * mid1 * mid3 * mid3 - 3 * mid3 * wid1 * wid1 - 3 * k2 * wid2 * wid2 - 6 * mid1 * wid2 * wid2 - 9 * mid3 * wid2 * wid2 - 3 * k3 * wid3 * wid3 - 6 * mid1 * wid3 * wid3 + 3 * mid1 * mid1 * mid3);
            vtkm::Float64 f4 = f * (-2 * mid1 * mid2 * mid2 * mid2 - 3 * mid3 * mid2 * mid2 * mid2 - 3 * mid1 * mid1 * mid2 * mid2 - 9 * mid3 * mid3 * mid2 * mid2 + 3 * wid1 * wid1 * mid2 * mid2 + 9 * wid3 * wid3 * mid2 * mid2 - 18 * mid1 * mid3 * mid2 * mid2 - 3 * mid3 * mid3 * mid3 * mid2 - 18 * mid1 * mid3 * mid3 * mid2 + 9 * mid3 * wid1 * wid1 * mid2 + 9 * k3 * wid3 * wid3 * mid2 + 18 * mid1 * wid3 * wid3 * mid2 - 9 * mid1 * mid1 * mid3 * mid2 - 2 * mid1 * mid3 * mid3 * mid3 - 3 * mid1 * mid1 * mid3 * mid3 + 3 * mid3 * mid3 * wid1 * wid1 + 3 * mid1 * mid1 * wid2 * wid2 + 9 * mid3 * mid3 * wid2 * wid2 - 3 * wid1 * wid1 * wid2 * wid2 + 6 * k2 * mid1 * wid2 * wid2 + 9 * k2 * mid3 * wid2 * wid2 + 18 * mid1 * mid3 * wid2 * wid2 + 3 * mid1 * mid1 * wid3 * wid3 - 3 * wid1 * wid1 * wid3 * wid3 - 9 * wid2 * wid2 * wid3 * wid3 + 6 * k3 * mid1 * wid3 * wid3);
            vtkm::Float64 f3 = f * (mid1 * mid1 * mid2 * mid2 * mid2 + 3 * mid3 * mid3 * mid2 * mid2 * mid2 - wid1 * wid1 * mid2 * mid2 * mid2 - 3 * wid3 * wid3 * mid2 * mid2 * mid2 + 6 * mid1 * mid3 * mid2 * mid2 * mid2 + 3 * mid3 * mid3 * mid3 * mid2 * mid2 + 18 * mid1 * mid3 * mid3 * mid2 * mid2 - 9 * mid3 * wid1 * wid1 * mid2 * mid2 - 9 * k3 * wid3 * wid3 * mid2 * mid2 - 18 * mid1 * wid3 * wid3 * mid2 * mid2 + 9 * mid1 * mid1 * mid3 * mid2 * mid2 + 6 * mid1 * mid3 * mid3 * mid3 * mid2 + 9 * mid1 * mid1 * mid3 * mid3 * mid2 - 9 * mid3 * mid3 * wid1 * wid1 * mid2 - 9 * mid1 * mid1 * wid3 * wid3 * mid2 + 9 * wid1 * wid1 * wid3 * wid3 * mid2 - 18 * k3 * mid1 * wid3 * wid3 * mid2 + mid1 * mid1 * mid3 * mid3 * mid3 - mid3 * mid3 * mid3 * wid1 * wid1 - 3 * mid3 * mid3 * mid3 * wid2 * wid2 - 3 * k2 * mid1 * mid1 * wid2 * wid2 - 9 * k2 * mid3 * mid3 * wid2 * wid2 - 18 * mid1 * mid3 * mid3 * wid2 * wid2 + 3 * k2 * wid1 * wid1 * wid2 * wid2 + 9 * mid3 * wid1 * wid1 * wid2 * wid2 - 9 * mid1 * mid1 * mid3 * wid2 * wid2 - 18 * k2 * mid1 * mid3 * wid2 * wid2 - 3 * k3 * mid1 * mid1 * wid3 * wid3 + 3 * k3 * wid1 * wid1 * wid3 * wid3 + 9 * k2 * wid2 * wid2 * wid3 * wid3 + 9 * k3 * wid2 * wid2 * wid3 * wid3 + 18 * mid1 * wid2 * wid2 * wid3 * wid3);
            vtkm::Float64 f2 = f * (-mid3 * mid3 * mid3 * mid2 * mid2 * mid2 - 6 * mid1 * mid3 * mid3 * mid2 * mid2 * mid2 + 3 * mid3 * wid1 * wid1 * mid2 * mid2 * mid2 + 3 * k3 * wid3 * wid3 * mid2 * mid2 * mid2 + 6 * mid1 * wid3 * wid3 * mid2 * mid2 * mid2 - 3 * mid1 * mid1 * mid3 * mid2 * mid2 * mid2 - 6 * mid1 * mid3 * mid3 * mid3 * mid2 * mid2 - 9 * mid1 * mid1 * mid3 * mid3 * mid2 * mid2 + 9 * mid3 * mid3 * wid1 * wid1 * mid2 * mid2 + 9 * mid1 * mid1 * wid3 * wid3 * mid2 * mid2 - 9 * wid1 * wid1 * wid3 * wid3 * mid2 * mid2 + 18 * k3 * mid1 * wid3 * wid3 * mid2 * mid2 - 3 * mid1 * mid1 * mid3 * mid3 * mid3 * mid2 + 3 * mid3 * mid3 * mid3 * wid1 * wid1 * mid2 + 9 * k3 * mid1 * mid1 * wid3 * wid3 * mid2 - 9 * k3 * wid1 * wid1 * wid3 * wid3 * mid2 + 3 * k2 * mid3 * mid3 * mid3 * wid2 * wid2 + 6 * mid1 * mid3 * mid3 * mid3 * wid2 * wid2 + 9 * mid1 * mid1 * mid3 * mid3 * wid2 * wid2 + 18 * k2 * mid1 * mid3 * mid3 * wid2 * wid2 - 9 * mid3 * mid3 * wid1 * wid1 * wid2 * wid2 - 9 * k2 * mid3 * wid1 * wid1 * wid2 * wid2 + 9 * k2 * mid1 * mid1 * mid3 * wid2 * wid2 - 9 * mid1 * mid1 * wid2 * wid2 * wid3 * wid3 + 9 * wid1 * wid1 * wid2 * wid2 * wid3 * wid3 - 9 * k2 * k3 * wid2 * wid2 * wid3 * wid3 - 18 * k2 * mid1 * wid2 * wid2 * wid3 * wid3 - 18 * k3 * mid1 * wid2 * wid2 * wid3 * wid3);
            vtkm::Float64 f1 = f * (2 * mid1 * mid3 * mid3 * mid3 * mid2 * mid2 * mid2 + 3 * mid1 * mid1 * mid3 * mid3 * mid2 * mid2 * mid2 - 3 * mid3 * mid3 * wid1 * wid1 * mid2 * mid2 * mid2 - 3 * mid1 * mid1 * wid3 * wid3 * mid2 * mid2 * mid2 + 3 * wid1 * wid1 * wid3 * wid3 * mid2 * mid2 * mid2 - 6 * k3 * mid1 * wid3 * wid3 * mid2 * mid2 * mid2 + 3 * mid1 * mid1 * mid3 * mid3 * mid3 * mid2 * mid2 - 3 * mid3 * mid3 * mid3 * wid1 * wid1 * mid2 * mid2 - 9 * k3 * mid1 * mid1 * wid3 * wid3 * mid2 * mid2 + 9 * k3 * wid1 * wid1 * wid3 * wid3 * mid2 * mid2 - 3 * mid1 * mid1 * mid3 * mid3 * mid3 * wid2 * wid2 - 6 * k2 * mid1 * mid3 * mid3 * mid3 * wid2 * wid2 - 9 * k2 * mid1 * mid1 * mid3 * mid3 * wid2 * wid2 + 3 * mid3 * mid3 * mid3 * wid1 * wid1 * wid2 * wid2 + 9 * k2 * mid3 * mid3 * wid1 * wid1 * wid2 * wid2 + 9 * k2 * mid1 * mid1 * wid2 * wid2 * wid3 * wid3 + 9 * k3 * mid1 * mid1 * wid2 * wid2 * wid3 * wid3 - 9 * k2 * wid1 * wid1 * wid2 * wid2 * wid3 * wid3 - 9 * k3 * wid1 * wid1 * wid2 * wid2 * wid3 * wid3 + 18 * k2 * k3 * mid1 * wid2 * wid2 * wid3 * wid3);
            vtkm::Float64 f0 = f * (-mid1 * mid1 * mid2 * mid2 * mid2 * mid3 * mid3 * mid3 + mid2 * mid2 * mid2 * mid3 * mid3 * mid3 * wid1 * wid1 + 3 * k2 * mid1 * mid1 * mid3 * mid3 * mid3 * wid2 * wid2 - 3 * k2 * mid3 * mid3 * mid3 * wid1 * wid1 * wid2 * wid2 + 3 * k3 * mid1 * mid1 * mid2 * mid2 * mid2 * wid3 * wid3 - 3 * k3 * mid2 * mid2 * mid2 * wid1 * wid1 * wid3 * wid3 - 9 * k2 * k3 * mid1 * mid1 * wid2 * wid2 * wid3 * wid3 + 9 * k2 * k3 * wid1 * wid1 * wid2 * wid2 * wid3 * wid3);

            // Below is plain multiplication of (k4 - x + (x-m4)^3/(3w4^2))*(k5 - x + (x-m5)^3/(3w5^2))
            // separated into factors f of x^6, x^5.....

            vtkm::Float64 p = 1.0 / (9.0 * wid4 * wid4 * wid5 * wid5);
            vtkm::Float64 p6 = p * 1;
            vtkm::Float64 p5 = p * (-3 * mid4 - 3 * mid5);
            vtkm::Float64 p4 = p * (3 * mid4 * mid4 + 9 * mid5 * mid4 + 3 * mid5 * mid5 - 3 * wid4 * wid4 - 3 * wid5 * wid5);
            vtkm::Float64 p3 = p * (3 * k4 * wid4 * wid4 + 3 * k5 * wid5 * wid5 + 9 * mid4 * wid5 * wid5 + 9 * mid5 * wid4 * wid4 - mid4 * mid4 - 9 * mid5 * mid4 * mid4 - 9 * mid5 * mid5 * mid4 - mid5 * mid5 * mid5);
            vtkm::Float64 p2 = p * (-9 * k5 * mid4 * wid5 * wid5 - 9 * k4 * mid5 * wid4 * wid4 - 9 * mid4 * mid4 * wid5 * wid5 - 9 * mid5 * mid5 * wid4 * wid4 + 3 * mid5 * mid4 * mid4 + 9 * mid5 * mid5 * mid4 * mid4 + 3 * mid5 * mid5 * mid5 * mid4 + 9 * wid4 * wid4 * wid5 * wid5);
            vtkm::Float64 p1 = p * (9 * k5 * mid4 * mid4 * wid5 * wid5 + 9 * k4 * mid5 * mid5 * wid4 * wid4 - 9 * k4 * wid4 * wid4 * wid5 * wid5 - 9 * k5 * wid4 * wid4 * wid5 * wid5 + 3 * mid4 * mid4 * wid5 * wid5 + 3 * mid5 * mid5 * mid5 * wid4 * wid4 - 3 * mid5 * mid5 * mid4 * mid4 - 3 * mid5 * mid5 * mid5 * mid4 * mid4);
            vtkm::Float64 p0 = p * (-3 * k4 * mid5 * mid5 * mid5 * wid4 * wid4 - 3 * k5 * mid4 * mid4 * wid5 * wid5 + 9 * k4 * k5 * wid4 * wid4 * wid5 * wid5 + mid4 * mid4 * mid5 * mid5 * mid5);

            // Below is the integral of (f8*x^8+f7*x^7+..+f0)*(p6*x^6+...p0) calculated and separated into
            // factors s of x^15, x^14, ..

            vtkm::Float64 t = (1.0 / 360360);
            vtkm::Float64 t15 = t * (24024 * f8 * p6);
            vtkm::Float64 t14 = t * (25740 * f7 * p6 + 25740 * f8 * p5);
            vtkm::Float64 t13 = t * (27720 * f6 * p6 + 27720 * f7 * p5 + 27720 * f8 * p4);
            vtkm::Float64 t12 = t * (30030 * f5 * p6 + 30030 * f6 * p5 + 30030 * f7 * p4 + 30030 * f8 * p3);
            vtkm::Float64 t11 = t * (32760 * f4 * p6 + 32760 * f5 * p5 + 32760 * f6 * p4 + 32760 * f7 * p3 + 32760 * f8 * p2);
            vtkm::Float64 t10 = t * (36036 * f3 * p6 + 36036 * f4 * p5 + 36036 * f5 * p4 + 36036 * f6 * p3 + 36036 * f7 * p2 + 36036 * f8 * p1);
            vtkm::Float64 t9 = t * (40040 * f2 * p6 + 40040 * f3 * p5 + 40040 * f4 * p4 + 40040 * f5 * p3 + 40040 * f6 * p2 + 40040 * f7 * p1 + 40040 * f8 * p0);
            vtkm::Float64 t8 = t * (45045 * f1 * p6 + 45045 * f2 * p5 + 45045 * f3 * p4 + 45045 * f4 * p3 + 45045 * f5 * p2 + 45045 * f6 * p1 + 45045 * f7 * p0);
            vtkm::Float64 t7 = t * (51480 * f0 * p6 + 51480 * f1 * p5 + 51480 * f2 * p4 + 51480 * f3 * p3 + 51480 * f4 * p2 + 51480 * f5 * p1 + 51480 * f6 * p0);
            vtkm::Float64 t6 = t * (60060 * f0 * p5 + 60060 * f1 * p4 + 60060 * f2 * p3 + 60060 * f3 * p2 + 60060 * f4 * p1 + 60060 * f5 * p0);
            vtkm::Float64 t5 = t * (72072 * f0 * p4 + 72072 * f1 * p3 + 72072 * f2 * p2 + 72072 * f3 * p1 + 72072 * f4 * p0);
            vtkm::Float64 t4 = t * (90090 * f0 * p3 + 90090 * f1 * p2 + 90090 * f2 * p1 + 90090 * f3 * p0);
            vtkm::Float64 t3 = t * (120120 * f0 * p2 + 120120 * f1 * p1 + 120120 * f2 * p0);
            vtkm::Float64 t2 = t * (180180 * f0 * p1 + 180180 * f1 * p0);
            vtkm::Float64 t1 = t * (360360 * f0 * p0);

            intUp = normalizingFactor * (t15 * vtkm::Pow(h, 15) + t14 * vtkm::Pow(h, 14) + t13 * vtkm::Pow(h, 13) + t12 * vtkm::Pow(h, 12) + t11 * vtkm::Pow(h, 11) + t10 * vtkm::Pow(h, 10) + t9 * vtkm::Pow(h, 9) + t8 * vtkm::Pow(h, 8) + t7 * vtkm::Pow(h, 7) + t6 * vtkm::Pow(h, 6) + t5 * vtkm::Pow(h, 5) + t4 * vtkm::Pow(h, 4) + t3 * vtkm::Pow(h, 3) + t2 * h * h + t1 * h);
            intDown = normalizingFactor * (t15 * vtkm::Pow(l, 15) + t14 * vtkm::Pow(l, 14) + t13 * vtkm::Pow(l, 13) + t12 * vtkm::Pow(l, 12) + t11 * vtkm::Pow(l, 11) + t10 * vtkm::Pow(l, 10) + t9 * vtkm::Pow(l, 9) + t8 * vtkm::Pow(l, 8) + t7 * vtkm::Pow(l, 7) + t6 * vtkm::Pow(l, 6) + t5 * vtkm::Pow(l, 5) + t4 * vtkm::Pow(l, 4) + t3 * vtkm::Pow(l, 3) + t2 * l * l + t1 * l);

            // print('intUp - intDown', intUp-intDown)
            if ((intUp - intDown) > 1)
                printf("error in integral 5");
        }

        return (intUp - intDown);
    }

private:
    double m_ScaleNum = 10000;
};

#endif // UCV_CRITICAL_POINT_EPANECH_AOUF_h